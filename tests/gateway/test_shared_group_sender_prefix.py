import asyncio
import json

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner(config: GatewayConfig) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


def test_preprocess_prefixes_sender_for_shared_non_thread_group_session():
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

    result = asyncio.run(
        runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )
    )

    assert result == "[Alice] hello"


def test_preprocess_keeps_plain_text_for_default_group_sessions():
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

    result = asyncio.run(
        runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )
    )

    assert result == "hello"


def test_preprocess_uses_platform_group_session_override():
    napcat = Platform("napcat")
    runner = _make_runner(
        GatewayConfig(
            platforms={
                napcat: PlatformConfig(
                    enabled=True,
                    extra={"group_sessions_per_user": False},
                ),
            },
            group_sessions_per_user=True,
        )
    )
    source = SessionSource(
        platform=napcat,
        chat_id="group:610066383",
        chat_name="Test Group",
        chat_type="group",
        user_id="alice",
        user_name="Alice",
    )
    event = MessageEvent(text="hello", source=source)

    result = asyncio.run(
        runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )
    )

    assert result == "[Alice] hello"


def test_preprocess_adds_qq_id_for_shared_qq_group_session():
    milky = Platform("milky")
    runner = _make_runner(
        GatewayConfig(
            platforms={
                milky: PlatformConfig(
                    enabled=True,
                    extra={"group_sessions_per_user": False},
                ),
            },
        )
    )
    source = SessionSource(
        platform=milky,
        chat_id="group:610066383",
        chat_name="小星群",
        chat_type="group",
        user_id="519434661",
        user_name="李泽铭",
    )
    event = MessageEvent(text="hello", source=source)

    result = asyncio.run(
        runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )
    )

    assert result == "[李泽铭|QQ:519434661] hello"


def test_preprocess_prefixes_known_person_alias_for_shared_group(monkeypatch):
    monkeypatch.setenv(
        "XIAOXING_PERSON_ALIASES",
        json.dumps([
            {
                "platform": "napcat",
                "external_id": "123456",
                "person_id": "alice",
                "person_name": "Alice Real",
            },
        ]),
    )
    napcat = Platform("napcat")
    runner = _make_runner(
        GatewayConfig(
            platforms={
                napcat: PlatformConfig(
                    enabled=True,
                    extra={"group_sessions_per_user": False},
                ),
            },
        )
    )
    source = SessionSource(
        platform=napcat,
        chat_id="group:610066383",
        chat_name="Test Group",
        chat_type="group",
        user_id="123456",
        user_name="Alice Card",
    )
    event = MessageEvent(text="hello", source=source)

    result = asyncio.run(
        runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )
    )

    assert result == "[Alice Real (Alice Card)|QQ:123456] hello"

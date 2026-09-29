import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.delivery import DeliveryTransport
from gateway.delivery_guard import HandoffDeliveryBlocked, guard_pre_delivery
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionEntry
from gateway.turn_context import TurnContext


def _handoff(body: str) -> str:
    return f"```text\n{body}\n```"


def _guard(content: str) -> None:
    guard_pre_delivery(
        platform="telegram",
        content=content,
        target={"chat_id": "fabricated"},
        session_id="handoff-test",
        turn_id="turn-test",
        metadata={"handoff": True},
    )


def test_telegram_handoff_exactly_2000_characters_is_allowed():
    content = _handoff("x" * (2_000 - len(_handoff(""))))

    assert len(content) == 2_000
    _guard(content)


def test_telegram_handoff_over_2000_characters_is_blocked_before_send():
    content = _handoff("x" * (2_001 - len(_handoff(""))))

    with pytest.raises(HandoffDeliveryBlocked, match=r"HANDOFF_DELIVERY_BLOCKED: 2001 > 2000"):
        _guard(content)


@pytest.mark.parametrize("content", [
    "handoff\n```text\nbody\n```",
    "```text\nbody\n```\nmore",
    "```python\nbody\n```",
    "```text\nbody\n```\n```text\nsecond\n```",
])
def test_telegram_handoff_requires_one_text_fence_only(content):
    with pytest.raises(HandoffDeliveryBlocked, match="HANDOFF_DELIVERY_BLOCKED: invalid fenced text block"):
        _guard(content)


def test_non_handoff_delivery_is_unchanged():
    guard_pre_delivery(
        platform="telegram", content="ordinary Telegram response that is not fenced",
        target={"chat_id": "fabricated"}, session_id="ordinary", turn_id="ordinary", metadata={},
    )


def test_handoff_transport_blocks_before_the_adapter_can_send_or_chunk():
    adapter = SimpleNamespace(send=AsyncMock())
    transport = DeliveryTransport(adapter, None, Platform.TELEGRAM)

    with pytest.raises(HandoffDeliveryBlocked):
        asyncio.run(transport.send(Platform.TELEGRAM, "fabricated", _handoff("x" * 2_000), {"handoff": True}))

    adapter.send.assert_not_awaited()


def test_final_delivery_retry_path_blocks_before_adapter_send_or_fallback():
    class Adapter(BasePlatformAdapter):
        async def connect(self, is_reconnect=False):
            raise NotImplementedError

        async def disconnect(self):
            raise NotImplementedError

        async def get_chat_info(self, _chat_id):
            raise NotImplementedError

        async def send(self, *_args, **_kwargs):
            raise NotImplementedError

    adapter = object.__new__(Adapter)
    adapter.platform = Platform.TELEGRAM
    adapter.send = AsyncMock()

    with pytest.raises(HandoffDeliveryBlocked):
        asyncio.run(BasePlatformAdapter._send_with_retry(
            adapter, "fabricated", _handoff("x" * 2_000), metadata={"handoff": True},
        ))

    adapter.send.assert_not_awaited()


def test_handoff_turn_never_creates_a_stream_consumer():
    adapter = SimpleNamespace(SUPPORTS_MESSAGE_EDITING=True)
    ctx = TurnContext(
        handoff_delivery=True,
        source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="fabricated"),
        user_config={},
        resolve_display_setting=lambda *_args: True,
        _run_still_current=lambda: True,
        interim_assistant_messages_enabled=True,
    )
    runner = SimpleNamespace(
        config=SimpleNamespace(streaming=SimpleNamespace(enabled_for=lambda _: True)),
        _delivery_adapter_for=lambda _source: adapter,
    )

    consumer, delta, interim, want_interim = TurnRunner(runner, ctx)._setup_stream_consumer("telegram")

    assert (consumer, delta, interim, want_interim) == (None, None, None, False)


def _make_handoff_runner(adapter):
    """GatewayRunner stand-in that resolves a REAL DeliveryTransport (so the delivery
    guard runs before adapter.send) against an obviously fabricated Telegram home."""
    runner = object.__new__(GatewayRunner)
    cfg = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
    )
    cfg.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM, chat_id="fabricated-target", name="fabricated home",
    )
    runner.config = cfg
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]), loaded_hooks=False,
    )
    runner._evict_cached_agent = MagicMock()
    runner._release_running_agent_state = MagicMock()
    runner._session_db = None

    store = MagicMock()
    store.get_or_create_session = AsyncMock(return_value=SessionEntry(
        session_key="k", session_id="s", created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="group",
    ))

    async def _switch(key, sid, *, preserve_prompt_pin=True):
        return SessionEntry(
            session_key=key, session_id=sid, created_at=datetime.now(), updated_at=datetime.now(),
            platform=Platform.TELEGRAM, chat_type="group",
        )

    store.switch_session = AsyncMock(side_effect=_switch)
    runner.session_store = store
    runner._async_session_store = SimpleNamespace(
        _store=store, get_or_create_session=store.get_or_create_session,
        switch_session=store.switch_session,
    )
    return runner


@pytest.mark.asyncio
async def test_handoff_rejected_payload_never_sent_and_one_valid_final_regenerated():
    """A guard-blocked Telegram handoff final must trigger exactly one bounded regeneration
    turn (whose prompt carries HANDOFF_DELIVERY_BLOCKED) and deliver only the valid final.

    The 2001-character payload is rejected by the guard BEFORE the adapter sees it; the
    regeneration turn produces a valid fenced block that is sent exactly once.
    """
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))
    adapter.create_handoff_thread = AsyncMock(return_value=None)
    adapter._bot = None
    runner = _make_handoff_runner(adapter)

    rejected = _handoff("x" * (2_001 - len(_handoff(""))))
    valid = _handoff("regenerated final")
    retry_texts = []
    responses = iter([rejected, valid])

    async def _handle_message(event):
        retry_texts.append(event.text)
        return next(responses)

    runner._handle_message = AsyncMock(side_effect=_handle_message)

    await runner._process_handoff({
        "id": "cli-session", "title": "work", "handoff_platform": "telegram",
    })

    # The regeneration prompt carried the machine-readable guard reason.
    assert any("HANDOFF_DELIVERY_BLOCKED" in text for text in retry_texts)
    # The rejected 2001-char payload never reached adapter.send.
    sent = [call.args[1] for call in adapter.send.await_args_list]
    assert rejected not in sent
    # Only the valid regenerated final was sent, exactly once.
    assert sent == [valid]
    assert adapter.send.await_count == 1

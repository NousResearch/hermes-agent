"""DM-only gateway status lines (#101003): the blocked-overflow compression warning reaches a DM and no shared chat.

Ported from #101011 (Wenfengcheng); the destination set is "anything but a DM" per the maintainer ruling.
"""

import pytest

from agent.conversation_compression import CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE
from hermes_agent.gateway.config import Platform
from hermes_agent.gateway.run import _prepare_gateway_status_message


@pytest.mark.parametrize(
    "chat_type", ["group", "forum", "channel", "supergroup", "thread", "guild"]
)
def test_blocked_overflow_warning_is_suppressed_in_multi_user_chats(chat_type):
    warning = CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(
        tokens=85_000, threshold=72_000, reason="cooldown:30"
    )
    assert (
        _prepare_gateway_status_message(
            Platform.TELEGRAM, "warn", warning, chat_type
        )
        is None
    )


@pytest.mark.parametrize("chat_type", ["group", "dm"])
def test_blocked_overflow_warning_through_production_callback(chat_type):
    import asyncio

    from hermes_agent.gateway.run_turn_runner import TurnRunner
    from hermes_agent.gateway.session import SessionSource
    from hermes_agent.gateway.turn_context import TurnContext

    warning = CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(
        tokens=85_000, threshold=72_000, reason="cooldown:30"
    )
    delivered = []

    class Transport:
        async def send(self, chat_id, text, metadata=None):
            delivered.append((chat_id, text, metadata))

    async def exercise():
        loop = asyncio.get_running_loop()
        ctx = TurnContext(
            source=SessionSource(platform=Platform.TELEGRAM, chat_id="destination", chat_type=chat_type),
            user_config={}, _run_still_current=lambda: True, _loop_for_step=loop,
        )
        ctx._status_adapter = Transport()
        ctx._status_chat_id = "destination"
        ctx._status_thread_metadata = {"thread_id": "topic"}
        runner = TurnRunner(None, ctx)

        def emit():
            runner._status_callback_sync("warn", warning)
            # Drain the real thread-to-loop scheduler before inspecting delivery.
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(timeout=5)

        await asyncio.to_thread(emit)

    asyncio.run(exercise())
    assert delivered == (
        [] if chat_type == "group" else
        [("destination", warning, {"thread_id": "topic"})]
    )


def test_blocked_overflow_warning_stays_visible_in_dm_and_raw_surfaces():
    warning = CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(
        tokens=85_000, threshold=72_000, reason="ineffective"
    )
    assert (
        _prepare_gateway_status_message(
            Platform.TELEGRAM, "warn", warning, "dm"
        )
        == warning
    )
    for platform in ("local", "api_server", "webhook", "msgraph_webhook"):
        assert (
            _prepare_gateway_status_message(
                platform, "warn", warning, "group"
            )
            == warning
        )


@pytest.mark.parametrize("affix", ["prefix ", " extra", "\nextra", "\n"])
def test_similar_blocked_overflow_text_is_not_suppressed(affix):
    warning = CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(
        tokens=85_000, threshold=72_000, reason="cooldown:30"
    )
    message = affix + warning if affix == "prefix " else warning + affix
    assert (
        _prepare_gateway_status_message(
            Platform.TELEGRAM, "warn", message, "group"
        )
        == message.strip()
    )


@pytest.mark.parametrize(
    "reason",
    ["structural_backoff:295", "plugin(reason)", "line one\nline two", "x" * 257],
)
def test_blocked_overflow_match_accepts_full_emitted_reason_domain(reason):
    warning = CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(
        tokens=85_000, threshold=72_000, reason=reason
    )
    assert (
        _prepare_gateway_status_message(
            Platform.TELEGRAM, "warn", warning, "group"
        )
        is None
    )


@pytest.mark.parametrize("tokens", ["1,,2", "07,200", "８５,０００", "1000"])
def test_malformed_token_counts_are_not_suppressed(tokens):
    warning = CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(
        tokens=85_000, threshold=72_000, reason="cooldown:30"
    ).replace("85,000", tokens)
    assert (
        _prepare_gateway_status_message(
            Platform.TELEGRAM, "warn", warning, "group"
        )
        == warning
    )


def test_credential_shaped_reason_is_suppressed_before_redaction():
    warning = CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(
        tokens=85_000,
        threshold=72_000,
        reason="plugin credential sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ123456",
    )
    assert (
        _prepare_gateway_status_message(
            Platform.TELEGRAM, "warn", warning, "group"
        )
        is None
    )

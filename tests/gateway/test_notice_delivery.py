from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="channel",
        user_id="U123",
        thread_id="111.222",
    )


def _make_dm_source() -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="D123",
        chat_type="dm",
        user_id="U123",
    )


def _make_whatsapp_group_source() -> SessionSource:
    return SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="1234567890-1600000000@g.us",
        chat_type="group",
        user_id="9720000000001@c.us",
    )


def _make_runner(extra=None, platform=Platform.SLACK):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            platform: PlatformConfig(enabled=True, token="***", extra=extra or {})
        }
    )
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="public-1"))
    adapter.send_private_notice = AsyncMock(return_value=SendResult(success=True, message_id="private-1"))
    runner.adapters = {platform: adapter}
    return runner, adapter


@pytest.mark.asyncio
async def test_deliver_platform_notice_uses_private_delivery_when_configured():
    runner, adapter = _make_runner(extra={"notice_delivery": "private"})

    await runner._deliver_platform_notice(_make_source(), "hello")

    adapter.send_private_notice.assert_awaited_once_with(
        "C123",
        "U123",
        "hello",
        # user_id rides along since the R3-5 per-turn identity stamp (Slack
        # sources carry their author in thread metadata for the connector's
        # chat.startStream recipient fields). Additive and harmless here.
        metadata={"thread_id": "111.222", "user_id": "U123"},
    )
    adapter.send.assert_not_awaited()


# ── Group-scope suppression: operator notices must never post publicly in a
# multi-user chat. In a group-bot deployment the first activation is usually
# a public group, so the home-channel /sethome prompt and credit notices
# leaked internal setup details to every member.


@pytest.mark.asyncio
async def test_group_public_notice_is_suppressed():
    """Default (public) delivery in a group scope stays in the logs."""
    runner, adapter = _make_runner()

    await runner._deliver_platform_notice(_make_source(), "📬 No home channel is set…")

    adapter.send.assert_not_awaited()
    adapter.send_private_notice.assert_not_awaited()


@pytest.mark.asyncio
async def test_whatsapp_group_public_notice_is_suppressed():
    """The live leak shape: a WhatsApp group session's first activation."""
    runner, adapter = _make_runner(platform=Platform.WHATSAPP)

    await runner._deliver_platform_notice(
        _make_whatsapp_group_source(), "📬 No home channel is set…"
    )

    adapter.send.assert_not_awaited()
    adapter.send_private_notice.assert_not_awaited()


@pytest.mark.parametrize("chat_type", ["", None, "channel", "thread", "supergroup"])
def test_unknown_or_empty_chat_type_is_group_scope(chat_type):
    """Only an explicit DM value may take the DM path — everything else
    fails closed, including an adapter that never set a chat type."""
    from gateway.run import _source_is_group_scope

    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type=chat_type,
        user_id="U123",
    )

    assert _source_is_group_scope(source) is True


@pytest.mark.asyncio
async def test_unknown_chat_type_notice_is_suppressed():
    """The fail-closed classification reaches delivery, not just the helper."""
    runner, adapter = _make_runner()
    source = SessionSource(
        platform=Platform.SLACK, chat_id="C123", chat_type="", user_id="U123"
    )

    await runner._deliver_platform_notice(source, "📬 No home channel is set…")

    adapter.send.assert_not_awaited()
    adapter.send_private_notice.assert_not_awaited()


@pytest.mark.asyncio
async def test_dm_public_notice_still_delivered():
    """DM delivery is unchanged — the 'public' chat IS the operator."""
    runner, adapter = _make_runner()

    await runner._deliver_platform_notice(_make_dm_source(), "hello")

    adapter.send.assert_awaited_once()
    assert adapter.send.call_args.args[0] == "D123"


@pytest.mark.asyncio
async def test_group_private_failure_does_not_fall_back_to_public():
    """A failed private send in a group must not degrade to a public post."""
    runner, adapter = _make_runner(extra={"notice_delivery": "private"})
    adapter.send_private_notice = AsyncMock(
        return_value=SendResult(success=False, error="no ephemeral")
    )

    await runner._deliver_platform_notice(_make_source(), "hello")

    adapter.send_private_notice.assert_awaited_once()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_dm_private_failure_still_falls_back_to_public():
    """The pre-existing DM fallback (private → public) is preserved."""
    runner, adapter = _make_runner(extra={"notice_delivery": "private"})
    adapter.send_private_notice = AsyncMock(
        return_value=SendResult(success=False, error="nope")
    )

    await runner._deliver_platform_notice(_make_dm_source(), "hello")

    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_group_private_base_default_does_not_count_as_private():
    """The BasePlatformAdapter send_private_notice default falls back to a
    normal public send — in a group that fallback IS the leak, so it must
    not be treated as a private path."""

    class _NoPrivatePathAdapter:
        # Deliberately the Base default, not an override.
        send_private_notice = BasePlatformAdapter.send_private_notice

        def __init__(self):
            self.send = AsyncMock(
                return_value=SendResult(success=True, message_id="public-1")
            )

    runner, _ = _make_runner(
        extra={"notice_delivery": "private"}, platform=Platform.WHATSAPP
    )
    adapter = _NoPrivatePathAdapter()
    runner.adapters = {Platform.WHATSAPP: adapter}

    await runner._deliver_platform_notice(_make_whatsapp_group_source(), "hello")

    adapter.send.assert_not_awaited()



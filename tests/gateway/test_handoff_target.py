"""Handoff target grammar is a strict subset of delivery targets."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.delivery import DeliveryTransport, parse_handoff_target


def test_bare_platform_uses_home_channel():
    target = parse_handoff_target("Slack")
    assert target.platform == Platform.SLACK
    assert target.chat_id is None
    assert target.to_string() == "slack"


def test_explicit_target_preserves_chat_id_case():
    target = parse_handoff_target("SLACK:C012MixedCase")
    assert target.platform == Platform.SLACK
    assert target.chat_id == "C012MixedCase"
    assert target.to_string() == "slack:C012MixedCase"


def test_matrix_room_id_preserves_internal_colon():
    target = parse_handoff_target("matrix:!room:example.org")
    assert target.platform == Platform.MATRIX
    assert target.chat_id == "!room:example.org"
    assert target.thread_id is None
    assert target.to_string() == "matrix:!room:example.org"


@pytest.mark.parametrize("value", ["local", "origin", "unknown:C1", "slack:"])
def test_non_messaging_or_incomplete_targets_are_rejected(value):
    with pytest.raises(ValueError):
        parse_handoff_target(value)


def test_existing_thread_target_is_rejected():
    with pytest.raises(ValueError, match="fresh thread"):
        parse_handoff_target("slack:C1:1700000000.000100")


@pytest.mark.asyncio
async def test_relay_thread_creation_preserves_logical_platform():
    adapter = AsyncMock()
    adapter.create_handoff_thread.return_value = "thread-1"
    transport = DeliveryTransport(adapter, None, Platform.RELAY)

    result = await transport.create_handoff_thread(
        Platform.DISCORD, "channel-1", "Hermes — task"
    )

    assert result == "thread-1"
    adapter.create_handoff_thread.assert_awaited_once_with(
        "channel-1", "Hermes — task", platform="discord", scope_id=None
    )


@pytest.mark.asyncio
async def test_native_slack_thread_creation_preserves_workspace_scope():
    adapter = AsyncMock()
    adapter.create_handoff_thread.return_value = "1700000000.1"
    transport = DeliveryTransport(adapter, None, Platform.SLACK)

    result = await transport.create_handoff_thread(
        Platform.SLACK, "C1", "Hermes — task", scope_id="T_WORKSPACE"
    )

    assert result == "1700000000.1"
    adapter.create_handoff_thread.assert_awaited_once_with(
        "C1", "Hermes — task", scope_id="T_WORKSPACE"
    )


@pytest.mark.asyncio
async def test_legacy_two_argument_slack_hook_remains_compatible():
    class LegacySlackAdapter:
        def __init__(self):
            self.calls = []

        async def create_handoff_thread(self, chat_id, name):
            self.calls.append((chat_id, name))
            return "thread"

    adapter = LegacySlackAdapter()
    transport = DeliveryTransport(adapter, None, Platform.SLACK)

    assert await transport.create_handoff_thread(Platform.SLACK, "C1", "Task") == "thread"
    assert await transport.create_handoff_thread(
        Platform.SLACK, "C1", "Scoped", scope_id="T1"
    ) is None
    assert adapter.calls == [("C1", "Task")]
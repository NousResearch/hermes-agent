"""Targeted tests for Desktop/TUI → gateway handoff destinations.

The legacy `/handoff slack` path must keep using Slack's configured home
channel, while `/handoff slack:pm` routes through a configured alias instead of
mutating the global home channel.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from gateway.config import (
    GatewayConfig,
    HomeChannel,
    Platform,
    PlatformConfig,
    parse_handoff_platform_ref,
    resolve_handoff_destination,
)
from gateway.run import GatewayRunner


PM_ID = "C_PM"
PMO_ID = "C_PMO"


def _slack_config() -> GatewayConfig:
    return GatewayConfig(
        platforms={
            Platform.SLACK: PlatformConfig(
                enabled=True,
                token="fake-token",
                home_channel=HomeChannel(
                    platform=Platform.SLACK,
                    chat_id=PMO_ID,
                    name="#pmo",
                ),
                extra={
                    "reply_in_thread": True,
                    "handoff_targets": {
                        "pm": {
                            "chat_id": PM_ID,
                            "name": "#pm",
                        }
                    },
                },
            )
        }
    )


class _FakeAdapter:
    def __init__(self) -> None:
        self.created_parent_chat_id = None
        self.created_thread_name = None
        self.sent = None

    async def create_handoff_thread(self, parent_chat_id: str, name: str):
        self.created_parent_chat_id = parent_chat_id
        self.created_thread_name = name
        return "1700000000.000001"

    async def send(self, chat_id: str, content: str, metadata=None):
        self.sent = {
            "chat_id": chat_id,
            "content": content,
            "metadata": metadata,
        }
        return SimpleNamespace(success=True)


class _FakeSessionStore:
    def __init__(self) -> None:
        self.created_source = None
        self.switch_call = None

    def get_or_create_session(self, source):
        self.created_source = source
        return SimpleNamespace(id="destination")

    def switch_session(self, session_key: str, cli_session_id: str):
        self.switch_call = (session_key, cli_session_id)
        return SimpleNamespace(id=cli_session_id)


def _runner(config: GatewayConfig, adapter: _FakeAdapter) -> Any:
    runner = cast(Any, GatewayRunner.__new__(GatewayRunner))
    runner.config = config
    runner.adapters = {Platform.SLACK: adapter}
    runner.session_store = _FakeSessionStore()
    runner.evicted = []
    runner.released = []
    runner.synthetic_event = None
    runner._evict_cached_agent = lambda session_key: runner.evicted.append(session_key)

    def _release_running_agent_state(session_key, *, run_generation=None):
        runner.released.append((session_key, run_generation))
        return True

    runner._release_running_agent_state = _release_running_agent_state

    async def _handle_message(event):
        runner.synthetic_event = event
        return "handoff ready"

    runner._handle_message = _handle_message
    return runner


def test_resolve_handoff_destination_defaults_to_home_channel():
    platform, home, alias = resolve_handoff_destination(_slack_config(), "slack")

    assert platform is Platform.SLACK
    assert alias is None
    assert home.chat_id == PMO_ID
    assert home.name == "#pmo"


def test_resolve_handoff_destination_uses_configured_alias():
    platform, home, alias = resolve_handoff_destination(_slack_config(), "slack:pm")

    assert platform is Platform.SLACK
    assert alias == "pm"
    assert home.chat_id == PM_ID
    assert home.name == "#pm"


def test_parse_handoff_platform_ref_rejects_raw_channel_targets():
    with pytest.raises(ValueError, match="invalid handoff target"):
        parse_handoff_platform_ref("slack:#pm")


@pytest.mark.asyncio
async def test_process_handoff_routes_legacy_slack_to_home_channel():
    adapter = _FakeAdapter()
    runner = _runner(_slack_config(), adapter)

    await GatewayRunner._process_handoff(
        runner,
        {"id": "cli-session", "title": "Desktop Session", "handoff_platform": "slack"},
    )

    assert adapter.created_parent_chat_id == PMO_ID
    assert adapter.sent["chat_id"] == PMO_ID
    assert runner.session_store.created_source.chat_id == PMO_ID
    assert runner.session_store.created_source.chat_name == "#pmo"
    assert runner.session_store.created_source.thread_id == "1700000000.000001"
    assert runner.session_store.switch_call[1] == "cli-session"


@pytest.mark.asyncio
async def test_process_handoff_routes_explicit_slack_pm_to_pm_target():
    adapter = _FakeAdapter()
    runner = _runner(_slack_config(), adapter)

    await GatewayRunner._process_handoff(
        runner,
        {"id": "cli-session", "title": "Desktop Session", "handoff_platform": "slack:pm"},
    )

    assert adapter.created_parent_chat_id == PM_ID
    assert adapter.sent["chat_id"] == PM_ID
    assert runner.session_store.created_source.chat_id == PM_ID
    assert runner.session_store.created_source.chat_name == "#pm"
    assert runner.session_store.created_source.thread_id == "1700000000.000001"
    assert runner.session_store.switch_call[1] == "cli-session"


@pytest.mark.asyncio
async def test_process_handoff_fails_unknown_configured_target():
    adapter = _FakeAdapter()
    runner = _runner(_slack_config(), adapter)

    with pytest.raises(RuntimeError, match="no handoff target 'missing'"):
        await GatewayRunner._process_handoff(
            runner,
            {"id": "cli-session", "title": "Desktop Session", "handoff_platform": "slack:missing"},
        )

    assert adapter.created_parent_chat_id is None

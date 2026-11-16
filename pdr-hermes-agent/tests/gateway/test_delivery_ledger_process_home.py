"""A served profile's reply is ledgered where the boot sweep looks for it.

A multiplexed gateway connects each served profile's adapter inside that profile's home override
(``_profile_runtime_scope``), and the receive loop the adapter starts while connecting inherits it,
so every final reply that bot sends is recorded from that context. The boot sweep that redelivers a
reply cut off by a crash runs in the launch context. The ledger used to resolve its path through
``get_hermes_home()``, which follows the override: the row landed in ``profiles/<name>/state.db``,
the sweep opened the launch ``state.db``, and the reply was never redelivered.

Every other ledger test replaces ``_db_path``; these leave it alone, since the path is what is
under test.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import hermes_constants
from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource

SESSION_KEY = "agent:research:slack:channel:C1"
ANSWER = "the final answer"


class _ServedAdapter(BasePlatformAdapter):
    """The served profile's own bot; its sends succeed, so the ledger is all that is observed."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self._owner_profile = "research"
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id="m-2")


@pytest.fixture(params=["hermes-home-env", "platform-default"])
def launch_home(request, tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "profiles" / "research").mkdir(parents=True)
    if request.param == "hermes-home-env":
        monkeypatch.setenv("HERMES_HOME", str(root))
    else:
        # A default gateway run in the foreground has no HERMES_HOME at all.
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: root)
    return root


def _runner(adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._profile_adapters = {"research": {Platform.SLACK: adapter}}
    runner._active_profile_name = lambda: "default"
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    runner.session_store = None
    runner._async_session_store = store
    return runner


@pytest.mark.asyncio
async def test_boot_sweep_redelivers_a_reply_recorded_under_a_served_profile_scope(launch_home):
    from gateway.run import _profile_runtime_scope

    adapter = _ServedAdapter()
    event = MessageEvent(
        text="hi", message_type=MessageType.TEXT, message_id="m-1",
        source=SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="channel"))
    # Stands in for the receive loop: a task created while the adapter connects under the scope.
    with _profile_runtime_scope(launch_home / "profiles" / "research", hydrate_secrets=False):
        record = asyncio.create_task(
            adapter._record_delivery_obligation(event, SESSION_KEY, ANSWER, adapter, False))
    assert await record

    # The gateway died after the send began and before the platform acknowledged it.
    with dl._connect() as conn:
        conn.execute("UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1")

    runner = _runner(adapter)
    assert await runner._redeliver_pending_obligations() == 1
    assert adapter.sent == [dl.RECOVERED_MARKER + ANSWER]
    runner._async_session_store.clear_resume_pending.assert_awaited_once_with(SESSION_KEY)

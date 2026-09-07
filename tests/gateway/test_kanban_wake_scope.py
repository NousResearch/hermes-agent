"""Kanban wake events must key to the same session as inbound messages.

Slack session keys include the workspace id, so the wake source the notifier
rebuilds from a subscription row must carry it too. The contract asserted here:
the key built from the wake source byte-matches the key built from an inbound
source for the same conversation, a scope-less key does not, and platforms
without tenant scoping keep their exact key shape.
"""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

from gateway.config import Platform, PlatformConfig
from gateway.kanban_watchers_notifier import _wake_scope_id
from gateway.run import GatewayRunner
from gateway.session import build_session_key
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from plugins.platforms.slack.adapter import SlackAdapter

TEAM = "T0B8U2M6NRE"
CHANNEL = "C0BCDG3H66P"
THREAD = "1720000000.000100"
USER = "U0BCE4NRVKN"


class UnscopedAdapter:
    """Push-capable adapter for a platform without tenant scoping."""

    def __init__(self):
        self.sent = []
        self.handled = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})

    async def handle_message(self, event):
        self.handled.append(event)


def _slack_adapter(channel_team=None):
    """Real SlackAdapter with only its I/O stubbed."""
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake-token"))
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._running = True
    adapter.send = AsyncMock()
    adapter.handle_message = AsyncMock()
    if channel_team:
        adapter._channel_team.update(channel_team)
    return adapter


def _runner(adapter, platform=Platform.SLACK):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {platform: adapter}
    runner._kanban_sub_fail_counts = {}
    # A gateway whose dispatcher owns the singleton lock.
    runner._kanban_dispatcher_lock_handle = object()
    return runner


async def _one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _completed_subscription(**sub_kwargs):
    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn,
            title="wake scope",
            assignee="worker",
            session_id="origin-session",
        )
        # Push-adapter wake injection is gated on the subscription's
        # delivery_mode ("notify+wake"/"wake") on current main; the plain
        # "notify" default would never reach the wake path under test.
        sub_kwargs.setdefault("delivery_mode", "notify+wake")
        kbn.add_notify_sub(conn, task_id=tid, **sub_kwargs)
        kb.complete_task(conn, tid, summary="done")
        return tid
    finally:
        conn.close()


def _wake_source_from(adapter):
    assert adapter.handle_message.await_count == 1, (
        f"expected exactly one wake injection, got {adapter.handle_message.await_count}"
    )
    return adapter.handle_message.await_args.args[0].source


def test_slack_wake_resumes_the_creators_workspace_scoped_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "wake-scope.db"))
    kb.init_db()
    _completed_subscription(
        platform="slack",
        chat_id=CHANNEL,
        chat_type="group",
        thread_id=THREAD,
        user_id=USER,
        # Slack sources are stamped with slack_team_id when subscribing.
        delivery_metadata={"slack_team_id": TEAM, "thread_id": THREAD},
    )

    adapter = _slack_adapter()
    asyncio.run(_one_notifier_tick(monkeypatch, _runner(adapter)))

    wake = _wake_source_from(adapter)
    assert wake.scope_id == TEAM

    inbound = adapter.build_source(
        chat_id=CHANNEL,
        chat_type="group",
        user_id=USER,
        thread_id=THREAD,
        scope_id=TEAM,
    )
    wake_key = build_session_key(wake)
    assert wake_key == build_session_key(inbound)
    assert TEAM in wake_key
    # A scope-less source keys to a different session for the same chat.
    assert build_session_key(replace(wake, scope_id=None, guild_id=None)) != wake_key


def test_slack_wake_falls_back_to_the_adapter_channel_workspace_map(tmp_path, monkeypatch):
    """Subscriptions that stored no workspace resolve it from the adapter."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "wake-scope-fallback.db"))
    kb.init_db()
    _completed_subscription(
        platform="slack",
        chat_id=CHANNEL,
        chat_type="group",
        thread_id=THREAD,
        delivery_metadata={"thread_id": THREAD, "chat_type": "group"},
    )

    adapter = _slack_adapter(channel_team={CHANNEL: TEAM})
    asyncio.run(_one_notifier_tick(monkeypatch, _runner(adapter)))

    wake = _wake_source_from(adapter)
    assert wake.scope_id == TEAM
    assert TEAM in build_session_key(wake)


def test_unknown_channel_keeps_the_previous_unscoped_wake(tmp_path, monkeypatch):
    """An unresolvable workspace yields an unscoped key, not a wrong scope."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "wake-scope-unknown.db"))
    kb.init_db()
    _completed_subscription(
        platform="slack",
        chat_id=CHANNEL,
        chat_type="group",
    )

    adapter = _slack_adapter()
    asyncio.run(_one_notifier_tick(monkeypatch, _runner(adapter)))

    wake = _wake_source_from(adapter)
    assert wake.scope_id is None
    assert build_session_key(wake) == f"agent:main:slack:group:{CHANNEL}"


def test_unscoped_platform_wake_key_is_byte_identical(tmp_path, monkeypatch):
    """Platforms without tenant scoping must keep their exact key shape."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "wake-scope-telegram.db"))
    kb.init_db()
    _completed_subscription(
        platform="telegram",
        chat_id="chat-dm",
        chat_type="dm",
    )

    adapter = UnscopedAdapter()
    asyncio.run(_one_notifier_tick(monkeypatch, _runner(adapter, Platform.TELEGRAM)))

    assert len(adapter.handled) == 1
    wake = adapter.handled[0].source
    assert wake.scope_id is None
    assert build_session_key(wake) == "agent:main:telegram:dm:chat-dm"


def test_wake_scope_id_prefers_persisted_metadata_over_the_adapter_map():
    """Persisted metadata wins; the adapter map is only a fallback."""
    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter._channel_team = {CHANNEL: "T_STALE"}

    assert _wake_scope_id(
        adapter, {"chat_id": CHANNEL, "delivery_metadata": {"slack_team_id": TEAM}}
    ) == TEAM
    assert _wake_scope_id(adapter, {"chat_id": CHANNEL}) == "T_STALE"
    assert _wake_scope_id(adapter, {"chat_id": "C_OTHER"}) is None


def test_wake_scope_id_degrades_when_the_adapter_lookup_raises():
    class Exploding:
        def scope_id_for_chat(self, chat_id):
            raise RuntimeError("adapter state gone")

    assert _wake_scope_id(Exploding(), {"chat_id": CHANNEL}) is None


def test_wake_scope_id_is_none_for_adapters_without_the_hook():
    """Adapters that don't resolve scopes leave the wake unscoped."""
    assert _wake_scope_id(UnscopedAdapter(), {"chat_id": CHANNEL}) is None


def test_slack_adapter_reports_the_channel_workspace():
    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter._channel_team = {CHANNEL: TEAM}

    assert adapter.scope_id_for_chat(CHANNEL) == TEAM
    assert adapter.scope_id_for_chat("C_UNKNOWN") is None
    assert adapter.scope_id_for_chat("") is None


def test_slack_adapter_reports_no_scope_for_ambiguous_channels():
    """A channel claimed by two workspaces resolves to no scope."""
    adapter = _slack_adapter()
    adapter._remember_channel_team("D_SHARED", "T_ONE")
    adapter._remember_channel_team("D_SHARED", "T_TWO")

    assert adapter.scope_id_for_chat("D_SHARED") is None

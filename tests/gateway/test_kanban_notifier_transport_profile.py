"""Kanban notifier routing regressions for multiplexed profile routes."""

from __future__ import annotations

import asyncio
import weakref
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import Platform
from gateway.profile_routing import ProfileRoute
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from gateway.session_context import get_session_env
from hermes_cli import kanban_db as kb
from tools.kanban_tools import _maybe_auto_subscribe


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append(
            {"chat_id": chat_id, "text": text, "metadata": metadata or {}}
        )


async def _run_one_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _runner(primary, *, secondary=None, configured=None):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: primary}
    runner._primary_profile_name = "default"
    runner._kanban_notifier_profile = "default"
    runner._kanban_dispatcher_lock_handle = object()
    runner._kanban_sub_fail_counts = {}
    runner._profile_adapters = {"daily": secondary or {}}
    runner._profile_configured_platforms = {
        "daily": set(configured or set()),
    }
    runner.config = SimpleNamespace(
        profile_routes=[
            ProfileRoute(
                name="daily-topic",
                platform="telegram",
                chat_id="-1003858887056",
                thread_id="2",
                profile="daily",
            )
        ]
    )
    return runner


def _completed_sub(*, notifier_profile="daily"):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="canary", assignee="dev")
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="-1003858887056",
            thread_id="2",
            notifier_profile=notifier_profile,
            delivery_mode="notify",
        )
        kb.complete_task(conn, task_id, summary="canary done")
        return task_id
    finally:
        conn.close()


def _subscription(task_id):
    conn = kb.connect()
    try:
        return kb.list_notify_subs(conn, task_id)[0]
    finally:
        conn.close()


def test_auto_subscribe_stamps_primary_transport_for_routed_secondary(
    tmp_path, monkeypatch,
):
    """A routed runtime profile must persist the bot/transport owner profile."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "auto-sub.db"))
    kb.init_db()

    primary = RecordingAdapter()
    runner = _runner(primary)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1003858887056",
        chat_type="group",
        thread_id="2",
        profile="daily",
    )
    source._transport_adapter_ref = weakref.ref(primary)
    context = SessionContext(
        source=source,
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
        session_key="agent:daily:telegram:group:-1003858887056:2",
    )

    tokens = runner._set_session_env(context)
    try:
        assert get_session_env("HERMES_SESSION_TRANSPORT_PROFILE") == "default"
        conn = kb.connect()
        try:
            task_id = kb.create_task(conn, title="child", assignee="dev")
            with patch("tools.kanban_tools.load_config", return_value={}):
                assert _maybe_auto_subscribe(conn, task_id) is True
        finally:
            conn.close()
    finally:
        runner._clear_session_env(tokens)

    assert _subscription(task_id)["notifier_profile"] == "default"


def test_virtual_routed_profile_replays_existing_subscription_once(
    tmp_path, monkeypatch,
):
    """Legacy runtime-stamped rows route through the matching primary bot."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "virtual.db"))
    kb.init_db()
    task_id = _completed_sub()
    before = _subscription(task_id)["last_event_id"]

    primary = RecordingAdapter()
    runner = _runner(primary)
    asyncio.run(_run_one_tick(monkeypatch, runner))

    assert len(primary.sent) == 1
    assert primary.sent[0]["chat_id"] == "-1003858887056"
    assert primary.sent[0]["metadata"]["thread_id"] == "2"
    after = _subscription(task_id)["last_event_id"]
    assert after > before

    runner._running = True
    asyncio.run(_run_one_tick(monkeypatch, runner))
    assert len(primary.sent) == 1
    assert _subscription(task_id)["last_event_id"] == after


def test_secondary_profile_with_distinct_bot_uses_its_own_adapter(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "distinct.db"))
    kb.init_db()
    _completed_sub()

    primary = RecordingAdapter()
    secondary = RecordingAdapter()
    runner = _runner(
        primary,
        secondary={Platform.TELEGRAM: secondary},
        configured={Platform.TELEGRAM},
    )
    asyncio.run(_run_one_tick(monkeypatch, runner))

    assert primary.sent == []
    assert len(secondary.sent) == 1


def test_secondary_profile_with_failed_adapter_never_falls_back_to_primary(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "failed.db"))
    kb.init_db()
    task_id = _completed_sub()
    before = _subscription(task_id)["last_event_id"]

    primary = RecordingAdapter()
    runner = _runner(primary, configured={Platform.TELEGRAM})
    asyncio.run(_run_one_tick(monkeypatch, runner))

    assert primary.sent == []
    assert _subscription(task_id)["last_event_id"] == before

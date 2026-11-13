"""A thread that lives inside its own chat does not need a ``parent_chat_id`` anchor.

A Slack DM thread records ``chat_id`` = the DM and ``thread_id`` = the message's timestamp: the
conversation IS the chat, so no parent is missing. The route gate used to stand a ROUTE's own
``chat_id`` in for that missing anchor, which let any channel route on the platform match itself
against every thread-shaped row — so an unrelated ``profile_routes`` entry refused a DM thread the
default profile owns (#151520). These cases pin the corrected treatment and the two neighbours it
must not disturb: a route that really does pin the DM still refuses, and a row whose ``chat_id`` is
the thread's own id (Discord) still cannot reach a channel-level route.
"""
import asyncio
from pathlib import Path

from hermes_agent.gateway.config import GatewayConfig, Platform
from hermes_agent.gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from hermes_agent.gateway.profile_routing import parse_profile_routes
from hermes_agent.gateway.run import GatewayRunner
from hermes_agent.hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn

DM = "D_TEST"
THREAD_TS = "1234567890.123456"
TEAM = "T_TEST"


class RecordingAdapter:
    supports_async_delivery = True

    def __init__(self):
        self.sent = []
        self.handled = []

    async def send(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text, kwargs))

    async def handle_message(self, event):
        self.handled.append(event)
        event._gateway_accepted = True


def setup_runner(tmp_path, monkeypatch, routes):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    for name in ("household-coordinator",):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.SLACK: RecordingAdapter()}
    runner._profile_adapters = {}
    runner._primary_profile_name = "default"
    runner._kanban_notifier_profile = "default"
    runner._kanban_dispatcher_lock_handle = object()
    runner.config = GatewayConfig(multiplex_profiles=True,
                                 profile_routes=parse_profile_routes(routes))
    return runner


def dm_thread_subscription(*, task_owner="default", metadata=None):
    """A default-owned Slack DM thread subscription with no ``parent_chat_id``."""
    with kbc.connect() as conn:
        task = kb.create_task(conn, title="dm thread completion", assignee="worker")
        kbn.add_notify_sub(conn, task_id=task, platform="slack", chat_id=DM,
                           thread_id=THREAD_TS, chat_type="dm", user_id="U_CREATOR",
                           notifier_profile=task_owner, delivery_mode="notify+wake",
                           delivery_metadata=metadata if metadata is not None else {"scope_id": TEAM})
        kb.complete_task(conn, task, result="finished")
    return task


def collect(runner):
    return _notifier_collect(runner, kb, notifier_profile="default", gc_due=False, gc_retention_days=30)


def unseen(task):
    with kbc.connect() as conn:
        return kbn.unseen_events_for_sub(conn, task_id=task, platform="slack", chat_id=DM,
                                         thread_id=THREAD_TS, kinds=["completed"])[1]


def test_dm_thread_delivers_past_an_unrelated_channel_route(tmp_path, monkeypatch):
    runner = setup_runner(tmp_path, monkeypatch, [
        dict(platform="slack", guild_id=TEAM, chat_id="C_OTHER", profile="household-coordinator"),
    ])
    task = dm_thread_subscription()
    rows = collect(runner)
    assert [row["task"].id for row in rows] == [task], "the unrelated channel route refused the DM thread"
    asyncio.run(_KanbanNotification(runner, rows[0], platform_cls=Platform, sub_fail_counts={}).deliver())
    adapter = runner.adapters[Platform.SLACK]
    assert [chat for chat, _text, _kw in adapter.sent] == [DM]
    assert adapter.sent[0][2]["metadata"]["thread_id"] == THREAD_TS
    assert not unseen(task)


def test_a_route_that_pins_the_dm_still_refuses(tmp_path, monkeypatch):
    runner = setup_runner(tmp_path, monkeypatch, [
        dict(platform="slack", guild_id=TEAM, chat_id=DM, profile="household-coordinator"),
    ])
    task = dm_thread_subscription()
    assert not collect(runner), "a route claiming the DM chat must still gate the subscription"
    assert unseen(task)


def test_a_thread_id_chat_still_needs_its_parent_anchor(tmp_path, monkeypatch):
    """A row whose ``chat_id`` is the thread's own id has no knowable parent, so the ordered gate
    keeps its fail-closed treatment: the later route that pins the row's own chat must not be
    reached past an earlier channel-level rule (#110919)."""
    runner = setup_runner(tmp_path, monkeypatch, [
        dict(platform="discord", guild_id=TEAM, chat_id="parent", profile="household-coordinator"),
        dict(platform="discord", guild_id=TEAM, chat_id="post", profile="household-coordinator"),
    ])
    runner.adapters = {Platform.DISCORD: RecordingAdapter()}
    with kbc.connect() as conn:
        task = kb.create_task(conn, title="discord thread completion", assignee="worker")
        kbn.add_notify_sub(conn, task_id=task, platform="discord", chat_id="post", thread_id="post",
                           chat_type="thread", user_id="U_CREATOR", notifier_profile="default",
                           delivery_mode="notify", delivery_metadata={"guild_id": TEAM})
        kb.complete_task(conn, task, result="finished")
    assert not collect(runner), "an anchorless thread-id row must stay fail-closed (#110919)"


def test_an_untyped_legacy_thread_row_keeps_the_unknown_parent_gate(tmp_path, monkeypatch):
    """A row predating ``chat_type`` cannot prove ``chat_id`` is the parent: row-level chat types
    arrive with the column, so an older Discord thread row may store the thread's own id there.
    The ordered gate keeps its historical treatment and an earlier channel route still refuses
    (#110919)."""
    runner = setup_runner(tmp_path, monkeypatch, [
        dict(platform="discord", guild_id=TEAM, chat_id="C_OTHER", profile="household-coordinator"),
    ])
    runner.adapters = {Platform.DISCORD: RecordingAdapter()}
    with kbc.connect() as conn:
        task = kb.create_task(conn, title="legacy thread completion", assignee="worker")
        kbn.add_notify_sub(conn, task_id=task, platform="discord", chat_id="post", thread_id="post",
                           chat_type="thread", user_id="U_CREATOR", notifier_profile="default",
                           delivery_mode="notify", delivery_metadata={"guild_id": TEAM})
        # Simulate a subscription stored before the chat_type column existed.
        conn.execute("UPDATE kanban_notify_subs SET chat_type = NULL WHERE task_id = ?", (task,))
        kb.complete_task(conn, task, result="finished")
    assert not collect(runner), "an untyped legacy row must not skip an earlier channel route"
    with kbc.connect() as conn:
        assert kbn.unseen_events_for_sub(conn, task_id=task, platform="discord", chat_id="post",
                                         thread_id="post", kinds=["completed"])[1]


def test_a_cli_thread_row_with_no_stated_chat_type_keeps_the_unknown_parent_gate(tmp_path, monkeypatch):
    """``notify-subscribe`` without ``--chat-type`` stores the column's 'dm' default; that default is
    no evidence ``chat_id`` is the thread's parent, so an earlier channel route still refuses."""
    import argparse
    from hermes_agent.hermes_cli import kanban as kc

    runner = setup_runner(tmp_path, monkeypatch, [
        dict(platform="discord", guild_id=TEAM, chat_id="C_OTHER", profile="household-coordinator"),
    ])
    runner.adapters = {Platform.DISCORD: RecordingAdapter()}
    with kbc.connect() as conn:
        task = kb.create_task(conn, title="cli thread completion", assignee="worker")
    parser = argparse.ArgumentParser()
    kc.build_parser(parser.add_subparsers(dest="command"))
    assert kc.kanban_command(parser.parse_args([
        "kanban", "notify-subscribe", task, "--platform", "discord", "--chat-id", "post",
        "--thread-id", "post", "--guild-id", TEAM, "--notifier-profile", "default"])) == 0
    with kbc.connect() as conn:
        kb.complete_task(conn, task, result="finished")
    assert not collect(runner), "a defaulted chat_type must not stand in for a known parent"

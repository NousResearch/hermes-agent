"""An explicitly owned board routes updates to its responsible conversation."""
import json
from pathlib import Path
from typing import Any

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from importlib import import_module


class _OwnerAPI:
    def __getattr__(self, name):
        return getattr(import_module("hermes_cli.kanban_notification_owner"), name)


kbo = _OwnerAPI()


OWNER: dict[str, Any] = dict(platform="local", chat_id="responsible-chat", notifier_profile="default",
             chat_type="dm", delivery_mode="wake")
OTHER: dict[str, Any] = dict(platform="local", chat_id="incidental-chat", notifier_profile="default",
             chat_type="dm", delivery_mode="wake")


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK",
                "HERMES_KANBAN_RUN_ID", "HERMES_SESSION_ID"):
        monkeypatch.delenv(key, raising=False)
    from gateway.session_context import set_session_vars, clear_session_vars
    session_tokens = set_session_vars(profile="default")
    kb._INITIALIZED_PATHS.clear()
    kb.create_board("project")
    try:
        with kbc.connect_closing(board="project") as conn:
            yield conn
    finally:
        clear_session_vars(session_tokens)


def test_owner_receives_new_tasks_without_a_creator_subscription(board):
    kbo.set_notification_owner(board, OWNER)
    task = kb.create_task(board, title="CLI-created", assignee="worker", session_id="creator")
    sub, = kbn.list_notify_subs(board)
    assert sub["task_id"] == task
    assert sub["chat_id"] == OWNER["chat_id"]
    assert sub["delivery_mode"] == "wake"
    assert kb.get_task(board, task).session_id == "creator"  # provenance is not ownership


def test_creator_cannot_divert_owner_notifications(board):
    kbo.set_notification_owner(board, OWNER)
    task = kb.create_task(board, title="foreign creator", assignee="worker")
    kbn.add_notify_sub(board, task_id=task, **OTHER)
    sub, = kbn.list_notify_subs(board)
    assert sub["chat_id"] == OWNER["chat_id"]
    assert sub["notifier_profile"] == OWNER["notifier_profile"]


def test_transfer_existing_tasks_preserves_pending_events(board):
    task = kb.create_task(board, title="existing", assignee="worker")
    kbn.add_notify_sub(board, task_id=task, **OTHER)
    before, = kbn.list_notify_subs(board)
    kb._append_event(board, task, "completed", {"result": "still pending"})
    board.commit()
    kbo.set_notification_owner(board, OWNER)
    sub, = kbn.list_notify_subs(board)
    assert sub["chat_id"] == OWNER["chat_id"]
    assert sub["last_event_id"] == before["last_event_id"]
    _, _, events = kbn.claim_unseen_events_for_sub(board, task_id=task, platform="local",
                                         chat_id=OWNER["chat_id"], kinds=("completed",))
    assert [event.payload["result"] for event in events] == ["still pending"]


def test_transfer_does_not_lose_events_claimed_but_not_delivered(board):
    task = kb.create_task(board, title="in flight", assignee="worker")
    kbn.add_notify_sub(board, task_id=task, **OTHER)
    kb._append_event(board, task, "completed", {"result": "not yet delivered"})
    board.commit()
    _, _, claimed = kbn.claim_unseen_events_for_sub(board, task_id=task, platform="local",
        chat_id=OTHER["chat_id"], kinds=("completed",))
    assert len(claimed) == 1
    kbo.set_notification_owner(board, OWNER)
    _, _, pending = kbn.claim_unseen_events_for_sub(board, task_id=task, platform="local",
        chat_id=OWNER["chat_id"], kinds=("completed",))
    assert [event.payload["result"] for event in pending] == ["not yet delivered"]


def test_transfer_does_not_replay_delivered_history(board):
    task = kb.create_task(board, title="settled", assignee="worker")
    kbn.add_notify_sub(board, task_id=task, **OTHER)
    kb._append_event(board, task, "completed", {"result": "already delivered"})
    board.commit()
    _, cursor, _ = kbn.claim_unseen_events_for_sub(board, task_id=task, platform="local", chat_id=OTHER["chat_id"],
                            kinds=("completed",))
    kbn.advance_notify_cursor(board, task_id=task, platform="local", chat_id=OTHER["chat_id"], new_cursor=cursor)
    kbo.set_notification_owner(board, OWNER)
    _, _, events = kbn.claim_unseen_events_for_sub(board, task_id=task, platform="local",
                                         chat_id=OWNER["chat_id"], kinds=("completed",))
    assert events == []


def test_owner_applies_to_parent_inheritance_and_late_subscriptions(board):
    parent = kb.create_task(board, title="parent", assignee="worker")
    kbn.add_notify_sub(board, task_id=parent, **OTHER)
    kbo.set_notification_owner(board, OWNER)
    child = kb.create_task(board, title="child", assignee="worker", parents=[parent])
    kbn.add_notify_sub(board, task_id=child, **OTHER)
    assert {sub["chat_id"] for sub in kbn.list_notify_subs(board)} == {OWNER["chat_id"]}


def test_unowned_board_retains_multiple_subscribers(board):
    task = kb.create_task(board, title="legacy", assignee="worker")
    assert kbo.get_notification_owner(board) is None
    kbn.add_notify_sub(board, task_id=task, **OWNER)
    kbn.add_notify_sub(board, task_id=task, **OTHER)
    assert {sub["chat_id"] for sub in kbn.list_notify_subs(board)} == {OWNER["chat_id"], OTHER["chat_id"]}


def test_clear_removes_owner_subscriptions_without_guessing_another_chat(board):
    kbo.set_notification_owner(board, OWNER)
    kb.create_task(board, title="owned", assignee="worker")
    kbo.set_notification_owner(board, None)
    assert kbo.get_notification_owner(board) is None
    assert kbn.list_notify_subs(board) == []


def test_owner_is_board_local(board):
    kbo.set_notification_owner(board, OWNER)
    kb.create_board("unrelated")
    with kbc.connect_closing(board="unrelated") as other:
        assert kbo.get_notification_owner(other) is None
        kb.create_task(other, title="unrelated", assignee="worker")
        assert kbn.list_notify_subs(other) == []


@pytest.mark.parametrize("target", [dict(OWNER, chat_id=""), dict(OWNER, notifier_profile=""),
                                         dict(OWNER, delivery_mode="typo"), dict(OWNER, platform="typo")])
def test_invalid_owner_is_rejected_without_changing_subscriptions(board, target):
    task = kb.create_task(board, title="existing", assignee="worker")
    kbn.add_notify_sub(board, task_id=task, **OTHER)
    with pytest.raises(ValueError):
        kbo.set_notification_owner(board, target)
    sub, = kbn.list_notify_subs(board)
    assert sub["chat_id"] == OTHER["chat_id"]


def test_cli_owner_configuration_and_readback(board, capsys):
    from hermes_cli import kanban as cli
    from hermes_cli.kanban_parser import build_parser
    import argparse
    root = argparse.ArgumentParser()
    build_parser(root.add_subparsers(dest="command"))
    def parse(argv):
        return root.parse_args(["kanban", *argv])
    from hermes_state import SessionDB
    import os
    state = SessionDB(Path(os.environ["HERMES_HOME"]) / "state.db")
    try:
        state.create_session(session_id="responsible-chat", source="tui", profile_name="default")
    finally:
        state.close()
    args = parse(["boards", "owner", "project", "--platform", "tui", "--chat-id", "responsible-chat",
                              "--profile", "default", "--json"])
    assert cli.kanban_command(args) == 0
    assert json.loads(capsys.readouterr().out)["notification_owner"]["chat_id"] == "responsible-chat"
    assert cli.kanban_command(parse(["boards", "owner", "project", "--json"])) == 0
    assert json.loads(capsys.readouterr().out)["notification_owner"]["platform"] == "tui"


def test_tool_reports_the_responsible_owner_not_the_incidental_creator(board, monkeypatch):
    from tools import kanban_tools as tools
    kbo.set_notification_owner(board, OWNER)
    monkeypatch.setattr(tools, "_resolve_notify_target", lambda: OTHER)
    result = json.loads(tools._handle_create(dict(title="different chat", assignee="worker", board="project")))
    assert result["ok"] is True
    assert result["notification_owner"]["chat_id"] == OWNER["chat_id"]
    sub, = kbn.list_notify_subs(board)
    assert sub["chat_id"] == OWNER["chat_id"]


def test_cli_unknown_session_is_not_silently_accepted(board, capsys):
    import argparse
    from hermes_cli import kanban as cli
    from hermes_cli.kanban_parser import build_parser
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["kanban", "boards", "owner", "project", "--platform", "local",
                              "--chat-id", "missing", "--profile", "default", "--json"])
    assert cli.kanban_command(args) != 0
    assert kbo.get_notification_owner(board) is None


def test_subscribe_again_preserves_cursor_and_ping_deduplication(board):
    kbo.set_notification_owner(board, OWNER)
    task = kb.create_task(board, title="dedup", assignee="worker")
    board.execute("UPDATE kanban_notify_subs SET last_ping_event_id=11, last_event_id=12 WHERE task_id=?", (task,))
    board.commit()
    kbn.add_notify_sub(board, task_id=task, **OTHER)
    sub, = kbn.list_notify_subs(board)
    assert (sub["last_event_id"], sub["last_ping_event_id"]) == (12, 11)


def test_stale_route_after_transfer_is_not_deliverable(board):
    kbo.set_notification_owner(board, OTHER)
    kb.create_task(board, title="transfer", assignee="worker")
    stale, = kbn.list_notify_subs(board)
    kbo.set_notification_owner(board, OWNER)
    assert not kbo.subscription_is_current(board, stale)
    current, = kbn.list_notify_subs(board)
    assert kbo.subscription_is_current(board, current)


def test_owner_route_metadata_is_preserved(board):
    target = dict(OWNER, platform="telegram", chat_id="owner-group", thread_id="topic-7",
                  chat_type="group", delivery_metadata={"scope_id": "workspace", "thread_id": "topic-7"})
    kbo.set_notification_owner(board, target)
    kb.create_task(board, title="topic", assignee="worker")
    sub, = kbn.list_notify_subs(board)
    assert sub["thread_id"] == "topic-7"
    assert sub["delivery_metadata"] == target["delivery_metadata"]


def test_tui_poller_delivers_only_to_responsible_conversation(board):
    from tui_gateway.session_notifications import _kb_poll_board
    kbo.set_notification_owner(board, dict(OWNER, platform="tui"))
    task = kb.create_task(board, title="owner E2E", assignee="worker", session_id=OTHER["chat_id"])
    kbn.add_notify_sub(board, task_id=task, **dict(OTHER, platform="tui"))
    kb.complete_task(board, task, summary="responsible only")
    assert _kb_poll_board(kb, "project", OTHER["chat_id"]) == []
    texts = _kb_poll_board(kb, "project", OWNER["chat_id"])
    assert len(texts) == 1 and "responsible only" in texts[0]
    assert _kb_poll_board(kb, "project", OWNER["chat_id"]) == []
    kbo.set_notification_owner(board, dict(OWNER, platform="tui", chat_id="replacement-chat"))
    assert _kb_poll_board(kb, "project", "replacement-chat") == []  # no replay


def test_gateway_notifier_delivers_only_to_responsible_conversation(board, monkeypatch):
    import asyncio
    from gateway.config import Platform
    from gateway.run import GatewayRunner

    target = dict(OWNER, platform="telegram", delivery_mode="notify")
    kbo.set_notification_owner(board, target)
    task = kb.create_task(board, title="gateway owner E2E", assignee="worker")
    kbn.add_notify_sub(board, task_id=task, **dict(OTHER, platform="telegram"))
    kb.complete_task(board, task, summary="gateway responsible only")
    class Adapter:
        def __init__(self):
            self.sent = []
        async def send(self, chat_id, text, metadata=None):
            self.sent.append((chat_id, text))
    adapter = Adapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    async def sleep(delay):
        if delay != 5:
            runner._running = False
    monkeypatch.setattr(asyncio, "sleep", sleep)
    asyncio.run(runner._kanban_notifier_watcher(interval=1))
    assert [chat for chat, text in adapter.sent] == [OWNER["chat_id"]]
    assert "gateway responsible only" in adapter.sent[0][1]


def test_gateway_drops_delivery_collected_before_owner_transfer(board, monkeypatch):
    import asyncio
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.kanban_watchers_notifier import _notifier_collect, _KanbanNotification
    target = dict(OTHER, platform="telegram", delivery_mode="notify")
    kbo.set_notification_owner(board, target)
    task = kb.create_task(board, title="stale snapshot", assignee="worker")
    kb.complete_task(board, task, summary="not for previous owner")
    class Adapter:
        async def send(self, *args, **kwargs):
            pytest.fail("stale owner must not receive this delivery")
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: Adapter()}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    deliveries = _notifier_collect(runner, kb, notifier_profile="default", gc_due=False, gc_retention_days=30)
    delivery, = [d for d in deliveries if d["sub"]["task_id"] == task]
    kbo.set_notification_owner(board, dict(target, chat_id=OWNER["chat_id"]))
    asyncio.run(_KanbanNotification(runner, delivery, platform_cls=Platform, sub_fail_counts={}).deliver())
    _, _, pending = kbn.claim_unseen_events_for_sub(board, task_id=task, platform="telegram",
        chat_id=OWNER["chat_id"], kinds=("completed",))
    assert len(pending) == 1


def test_reapplying_owner_does_not_reset_an_inflight_claim(board):
    kbo.set_notification_owner(board, OWNER)
    task = kb.create_task(board, title="idempotent ownership", assignee="worker")
    kb.complete_task(board, task, summary="in flight")
    kbn.claim_unseen_events_for_sub(board, task_id=task, platform="local", chat_id=OWNER["chat_id"], kinds=("completed",))
    before, = kbn.list_notify_subs(board)
    kbo.set_notification_owner(board, OWNER)
    after, = kbn.list_notify_subs(board)
    assert after == before


def test_string_and_decoded_delivery_metadata_match(board):
    target = dict(OWNER, delivery_metadata={"thread_id": "thread"})
    kbo.set_notification_owner(board, target)
    kb.create_task(board, title="metadata", assignee="worker")
    sub, = kbn.list_notify_subs(board)
    serialized = dict(sub, delivery_metadata=json.dumps(sub["delivery_metadata"]))
    assert kbo.subscription_matches_owner(serialized, sub)


def test_unowned_board_keeps_mutable_legacy_delivery_fields(board):
    task = kb.create_task(board, title="legacy re-subscribe", assignee="worker")
    kbn.add_notify_sub(board, task_id=task, **OTHER)
    old, = kbn.list_notify_subs(board)
    kbn.add_notify_sub(board, task_id=task, **dict(OTHER, delivery_mode="notify"))
    assert kbo.board_subscription_is_current("project", old)


def test_clearing_owner_rejects_previously_collected_delivery(board):
    kbo.set_notification_owner(board, OWNER)
    kb.create_task(board, title="clear owner", assignee="worker")
    stale, = kbn.list_notify_subs(board)
    kbo.set_notification_owner(board, None)
    assert not kbo.board_subscription_is_current("project", stale)


def test_export_scrubs_machine_local_owner(board):
    from hermes_cli.kanban_transfer import _scrub_local_state
    kbo.set_notification_owner(board, OWNER)
    with kb.write_txn(board):
        _scrub_local_state(board)
    assert kbo.get_notification_owner(board) is None

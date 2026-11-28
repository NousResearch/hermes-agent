"""Kanban completion pings the original turn actor, independently of its session route."""
import argparse
import asyncio

import pytest

from gateway.config import Platform
from hermes_cli import kanban, kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn
from tests.gateway.test_kanban_notifier import RecordingAdapter, _make_runner, _run_one_notifier_tick


@pytest.mark.parametrize("via_cli", [False, True])
def test_completion_keeps_original_actor_after_resubscribe(tmp_path, monkeypatch, via_cli):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    for key, value in {"PLATFORM": "discord", "CHAT_ID": "333", "USER_ID": "999",
                       "ACTOR_ID": "111", "CHAT_TYPE": "group"}.items():
        monkeypatch.setenv(f"HERMES_SESSION_{key}", value)
    kb.init_db()
    with kbc.connect_closing() as conn:
        task = kb.create_task(conn, title="Deliver result", assignee="worker")

    def subscribe():
        if via_cli:
            parser = argparse.ArgumentParser()
            kanban.build_parser(parser.add_subparsers(dest="command"))
            args = parser.parse_args(["kanban", "notify-subscribe", task, "--platform", "discord",
                                      "--chat-id", "333", "--chat-type", "group"])
            assert kanban.kanban_command(args) == 0
        else:
            from tools.kanban_tools import _resolve_notify_target
            with kbc.connect_closing() as conn:
                kbn.add_notify_sub(conn, task_id=task, **_resolve_notify_target())

    subscribe()
    monkeypatch.setenv("HERMES_SESSION_ACTOR_ID", "222")
    subscribe()
    with kbc.connect_closing() as conn:
        sub = kbn.list_notify_subs(conn, task)[0]
        assert sub["user_id"] == "999"  # Route identity is deliberately different.
        assert sub["delivery_metadata"]["requester_user_id"] == "111"
        kb.complete_task(conn, task, summary="Done <@111> <@!111> with help from <@222>")
        legacy = kb.create_task(conn, title="Existing subscription", assignee="worker")
        kbn.add_notify_sub(conn, task_id=legacy, platform="discord", chat_id="333")
        kbn.add_notify_sub(conn, task_id=legacy, platform="discord", chat_id="333",
                           delivery_metadata={"requester_user_id": "222", "guild_id": "777"})
        legacy_metadata = kbn.list_notify_subs(conn, legacy)[0]["delivery_metadata"]
        assert legacy_metadata == {"guild_id": "777"}
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner.adapters = {Platform.DISCORD: adapter}
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(adapter.sent) == 1
    ping = adapter.sent[0]
    assert ping["text"].startswith("<@111>\n")
    assert ping["text"].count("<@111>") == 1
    assert "<@!111>" not in ping["text"]
    assert ping["metadata"]["mention_user_ids"] == ["111"]
    runner._running = True
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(adapter.sent) == 1

    from hermes_cli.kanban_notify_thread import discord_requester_metadata
    assert discord_requester_metadata("discord", "555", None) == {}
    assert discord_requester_metadata("discord", "333", "555") == {}
    assert discord_requester_metadata("telegram", "333", None) == {}

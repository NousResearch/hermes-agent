"""A Telegram forum-topic subscription is routed by the group it lives in.

A card filed from a forum topic stores ``chat_id`` = the supergroup, ``thread_id`` = the topic and
``chat_type="group"``, with no ``parent_chat_id``: the topic lives inside the chat, so nothing is
missing. Under ``multiplex_profiles`` the notifier's route gate used to stand the FIRST chat-level
route's own chat in for the absent anchor, so another team's route matched and refused the row
forever (#148953). Only the route that names the topic's own group may claim it.
"""
import asyncio
from pathlib import Path

from hermes_agent.gateway.config import GatewayConfig, Platform
from hermes_agent.gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from hermes_agent.gateway.profile_routing import parse_profile_routes
from hermes_agent.gateway.run import GatewayRunner
from hermes_agent.hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn

GROUP_A = "-1001111111111"
GROUP_B = "-1002222222222"
TOPIC = "77"


class RecordingAdapter:
    supports_async_delivery = True

    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text, kwargs))


def _runner(tmp_path, monkeypatch, routes):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    for name in ("teama", "teamb"):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: RecordingAdapter()}
    runner._profile_adapters = {}
    runner._primary_profile_name = "default"
    runner._kanban_notifier_profile = "default"
    runner._kanban_dispatcher_lock_handle = object()
    runner.config = GatewayConfig(multiplex_profiles=True, profile_routes=parse_profile_routes(routes))
    return runner


def _topic_subscription(owner):
    """What ``kanban_create`` auto-subscribe stores from a Telegram forum topic."""
    with kbc.connect() as conn:
        task = kb.create_task(conn, title="topic card", assignee="worker")
        kbn.add_notify_sub(conn, task_id=task, platform="telegram", chat_id=GROUP_A, thread_id=TOPIC,
                           chat_type="group", user_id="111", notifier_profile=owner,
                           delivery_mode="notify", delivery_metadata={"chat_type": "group", "thread_id": TOPIC})
        kb.complete_task(conn, task, result="finished")
    return task


def _collect(runner):
    return _notifier_collect(runner, kb, notifier_profile="default", gc_due=False, gc_retention_days=30)


def test_topic_card_reaches_its_group_past_another_teams_earlier_route(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, [
        dict(platform="telegram", chat_id=GROUP_B, profile="teamb"),
        dict(platform="telegram", chat_id=GROUP_A, profile="teama"),
    ])
    task = _topic_subscription("teama")
    rows = _collect(runner)
    assert [row["task"].id for row in rows] == [task], "team B's route swallowed team A's topic card"
    asyncio.run(_KanbanNotification(runner, rows[0], platform_cls=Platform, sub_fail_counts={}).deliver())
    sent = runner.adapters[Platform.TELEGRAM].sent
    assert [chat for chat, _text, _kw in sent] == [GROUP_A]
    assert sent[0][2]["metadata"]["thread_id"] == TOPIC


def test_topic_card_owned_by_the_wrong_profile_still_refuses(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, [
        dict(platform="telegram", chat_id=GROUP_A, profile="teama"),
    ])
    _topic_subscription("teamb")
    assert not _collect(runner), "the route pinning the topic's group must still gate other owners"

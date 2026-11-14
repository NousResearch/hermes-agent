"""``kanban notify-subscribe`` without ``--thread-id`` keeps the ping in the topic the operator watches.

An omitted ``--thread-id`` used to record the chat's root lane, so a task dispatched from a Telegram
DM topic pinged the DM root and the subscription looked delivered.
"""
import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn

DM = "8631530276"


@pytest.fixture
def task(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in ("PLATFORM", "CHAT_ID", "THREAD_ID", "USER_ID", "USER_ID_ALT"):
        monkeypatch.delenv(f"HERMES_SESSION_{name}", raising=False)
    kb.init_db()
    with kbc.connect() as conn:
        return kb.create_task(conn, title="dm topic card", assignee="worker")


def _subscribe(*argv):
    parser = argparse.ArgumentParser()
    kc.build_parser(parser.add_subparsers(dest="command"))
    assert kc.kanban_command(parser.parse_args(["kanban", "notify-subscribe", *argv])) == 0


def _threads(task):
    with kbc.connect() as conn:
        return sorted(sub["thread_id"] for sub in kbn.list_notify_subs(conn, task))


def test_a_subscribe_from_the_topic_session_targets_that_topic(task, monkeypatch, capsys):
    for name, value in (("PLATFORM", "telegram"), ("CHAT_ID", DM), ("THREAD_ID", "42"), ("USER_ID", DM)):
        monkeypatch.setenv(f"HERMES_SESSION_{name}", value)
    _subscribe(task, "--platform", "telegram", "--chat-id", DM)
    assert _threads(task) == ["42"]
    assert "Subscribed telegram:8631530276:42" in capsys.readouterr().out


def test_an_existing_threaded_subscription_is_inherited_deterministically(task, capsys):
    with kbc.connect() as conn:
        for topic in ("topic2", "topic1"):
            kbn.add_notify_sub(conn, task_id=task, platform="telegram", chat_id=DM, thread_id=topic)
    _subscribe(task, "--platform", "telegram", "--chat-id", DM)
    err = capsys.readouterr().err
    assert "inherited thread_id=topic1" in err and "others: topic2" in err
    assert _threads(task) == ["topic1", "topic2"]  # no root-lane row added


def test_a_dm_with_nothing_to_infer_warns_and_an_explicit_thread_is_kept(task, capsys):
    _subscribe(task, "--platform", "telegram", "--chat-id", DM)
    assert "DM root" in capsys.readouterr().err
    assert _threads(task) == [""]
    with kbc.connect() as conn:
        other = kb.create_task(conn, title="explicit", assignee="worker")
    _subscribe(other, "--platform", "telegram", "--chat-id", DM, "--thread-id", "7")
    assert _threads(other) == ["7"]
    assert "warning" not in capsys.readouterr().err

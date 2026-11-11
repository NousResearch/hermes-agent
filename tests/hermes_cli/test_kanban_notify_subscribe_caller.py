"""``hermes kanban notify-subscribe`` run from a gateway session records the caller as the sender.

An agent in a Telegram group runs the command through its terminal tool without ``--user-id``.
Under ``group_sessions_per_user`` (the default) the wake rebuilds the session key from the row,
and a NULL ``user_id`` keys ``…:group:<chat>`` instead of the participant's
``…:group:<chat>:<user>``: a second session (#147522). The calling session's sender fills the
gap only when the subscription targets that same chat.
"""
import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn

GROUP = "-1003333333333"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        task = kb.create_task(conn, title="group card", assignee="worker")
    # What the terminal tool hands the CLI child of a group-chat turn (tools/environments/local.py).
    for name, value in (("PLATFORM", "telegram"), ("CHAT_ID", GROUP), ("CHAT_TYPE", "group"),
                        ("USER_ID", "111")):
        monkeypatch.setenv(f"HERMES_SESSION_{name}", value)
    return task


def _subscribe(*argv):
    parser = argparse.ArgumentParser()
    kc.build_parser(parser.add_subparsers(dest="command"))
    assert kc.kanban_command(parser.parse_args(["kanban", "notify-subscribe", *argv])) == 0


def _row(task):
    with kbc.connect() as conn:
        return kbn.list_notify_subs(conn, task)[0]


def test_subscribing_the_callers_own_group_records_the_caller(kanban_home):
    _subscribe(kanban_home, "--platform", "telegram", "--chat-id", GROUP, "--chat-type", "group",
               "--delivery-mode", "notify+wake")
    assert _row(kanban_home)["user_id"] == "111"


def test_another_chat_and_an_explicit_id_are_left_alone(kanban_home):
    _subscribe(kanban_home, "--platform", "telegram", "--chat-id", "-100999", "--chat-type", "group")
    assert _row(kanban_home)["user_id"] is None
    with kbc.connect() as conn:
        other = kb.create_task(conn, title="explicit", assignee="worker")
    _subscribe(other, "--platform", "telegram", "--chat-id", GROUP, "--user-id", "222")
    assert _row(other)["user_id"] == "222"

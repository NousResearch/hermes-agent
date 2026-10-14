"""A Discord subscription stored with the channel directory's "channel" label must wake the
chat's own "group" session, not a separate one."""

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "route.db"))
    kb.init_db()
    c = kbc.connect()
    yield c
    c.close()


def _sub(conn, **kwargs):
    tid = kb.create_task(conn, title="route", assignee="worker")
    kbn.add_notify_sub(conn, task_id=tid, platform=kwargs.pop("platform", "discord"), chat_id="123", **kwargs)
    return tid, kbn.list_notify_subs(conn, tid)[0]


def test_discord_channel_label_is_stored_as_group(conn):
    assert _sub(conn, chat_type="channel")[1]["chat_type"] == "group"


def test_other_platforms_keep_channel(conn):
    assert _sub(conn, platform="telegram", chat_type="channel")[1]["chat_type"] == "channel"


def test_resubscribing_with_the_label_keeps_the_group_session(conn):
    tid, _ = _sub(conn, chat_type="group")
    kbn.add_notify_sub(conn, task_id=tid, platform="discord", chat_id="123", chat_type="channel",
                       delivery_mode="notify+wake")
    assert kbn.list_notify_subs(conn, tid)[0]["chat_type"] == "group"


@pytest.mark.parametrize("platform, chat_type, expected", [
    ("discord", "channel", "group"), ("Discord", " Channel ", "group"), ("discord", "thread", "thread"),
    ("discord", "dm", "dm"), ("discord", None, None), ("slack", "channel", "channel"),
])
def test_session_chat_type(platform, chat_type, expected):
    assert kbn.session_chat_type(platform, chat_type) == expected

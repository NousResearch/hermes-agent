import json
import re
from pathlib import Path

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from gateway.slash_commands import GatewaySlashCommandsMixin
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


@pytest.fixture
def kanban_home_with_telegram_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for env_name in (
        "TELEGRAM_HOME_CHANNEL",
        "TELEGRAM_HOME_CHANNEL_THREAD_ID",
        "TELEGRAM_CRON_THREAD_ID",
    ):
        monkeypatch.delenv(env_name, raising=False)
    (home / "config.yaml").write_text(
        """
platforms:
  telegram:
    enabled: true
    home_channel:
      platform: telegram
      chat_id: tg-private-home
      name: Private Telegram
      thread_id: "42"
      user_id: tg-private-user
""".strip()
        + "\n",
        encoding="utf-8",
    )
    kb.init_db()
    return home


def _sub_by_platform(subs, platform):
    matches = [sub for sub in subs if sub["platform"] == platform]
    assert len(matches) == 1
    return matches[0]


class _KanbanSlashRunner(GatewaySlashCommandsMixin):
    def _reply_anchor_for_event(self, event):
        return None

    def _thread_metadata_for_source(self, source, anchor):
        metadata = {}
        if source.thread_id:
            metadata["thread_id"] = str(source.thread_id)
        if source.chat_type:
            metadata["chat_type"] = str(source.chat_type)
        return metadata

    def _active_profile_name(self):
        return "molly"


@pytest.mark.asyncio
async def test_gateway_kanban_create_from_imessage_splits_wake_and_raw_notify(
    kanban_home_with_telegram_home,
):
    """/kanban create from iMessage makes Photon/BlueBubbles wake-only and routes raw pings to Telegram."""
    source = SessionSource(
        platform=Platform.BLUEBUBBLES,
        chat_id="imessage-chat",
        chat_type="dm",
        user_id="imessage-user",
        thread_id="imessage-thread",
    )
    event = MessageEvent(
        text="/kanban create 'policy routed task' --assignee stark",
        source=source,
    )

    output = await _KanbanSlashRunner()._handle_kanban_command(event)
    match = re.search(r"Created\s+(t_[0-9a-f]+)\b", output)
    assert match, output
    task_id = match.group(1)

    with kbc.connect_closing() as conn:
        subs = kbn.list_notify_subs(conn, task_id)

    assert len(subs) == 2
    source_sub = _sub_by_platform(subs, "bluebubbles")
    assert source_sub["chat_id"] == "imessage-chat"
    assert source_sub["thread_id"] == "imessage-thread"
    assert source_sub["user_id"] == "imessage-user"
    assert source_sub["notifier_profile"] == "molly"
    assert source_sub["delivery_mode"] == "wake"

    raw_sub = _sub_by_platform(subs, "telegram")
    assert raw_sub["chat_id"] == "tg-private-home"
    assert raw_sub["thread_id"] == "42"
    assert raw_sub["user_id"] == "tg-private-user"
    assert raw_sub["notifier_profile"] == "molly"
    assert raw_sub["delivery_mode"] == "notify"
    assert raw_sub["delivery_metadata"]["telegram_dm_topic_reply_fallback"] is True


def test_kanban_create_tool_from_photon_splits_and_children_inherit_readback(
    kanban_home_with_telegram_home,
):
    """kanban_create auto-subscribe applies the same policy and child tasks inherit both rows."""
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.kanban_tools import _handle_create

    tokens = set_session_vars(
        platform="photon",
        chat_id="photon-chat",
        chat_type="dm",
        thread_id="photon-thread",
        user_id="photon-user",
        profile="molly",
    )
    try:
        result = json.loads(_handle_create({"title": "tool policy root", "assignee": "stark"}))
        assert result["ok"] is True
        assert result["subscribed"] is True
        root = result["task_id"]
        with kbc.connect_closing() as conn:
            child = kb.create_task(
                conn,
                title="tool policy child",
                assignee="reviewer",
                parents=[root],
            )
            root_subs = kbn.list_notify_subs(conn, root)
            child_subs = kbn.list_notify_subs(conn, child)
    finally:
        clear_session_vars(tokens)

    assert len(root_subs) == 2
    assert _sub_by_platform(root_subs, "photon")["delivery_mode"] == "wake"
    assert _sub_by_platform(root_subs, "telegram")["delivery_mode"] == "notify"

    assert len(child_subs) == 2
    child_source = _sub_by_platform(child_subs, "photon")
    child_raw = _sub_by_platform(child_subs, "telegram")
    assert child_source["chat_id"] == "photon-chat"
    assert child_source["thread_id"] == "photon-thread"
    assert child_source["delivery_mode"] == "wake"
    assert child_raw["chat_id"] == "tg-private-home"
    assert child_raw["thread_id"] == "42"
    assert child_raw["delivery_mode"] == "notify"


def test_auto_subscribe_preserves_existing_explicit_source_subscription(
    kanban_home_with_telegram_home,
):
    """Auto-subscribe may add the raw Telegram pair, but must not rewrite an existing source row."""
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.kanban_tools import _maybe_auto_subscribe

    tokens = set_session_vars(
        platform="photon",
        chat_id="photon-chat",
        chat_type="dm",
        thread_id="photon-thread",
        user_id="photon-user",
        profile="molly",
    )
    try:
        with kbc.connect_closing() as conn:
            task_id = kb.create_task(conn, title="explicit source preserved", assignee="stark")
            kbn.add_notify_sub(
                conn,
                task_id=task_id,
                platform="photon",
                chat_id="photon-chat",
                chat_type="dm",
                thread_id="photon-thread",
                user_id="explicit-user",
                notifier_profile="operator",
                delivery_mode="notify+wake",
                delivery_metadata={"explicit": "kept"},
            )
            assert _maybe_auto_subscribe(conn, task_id) is True
            subs = kbn.list_notify_subs(conn, task_id)
    finally:
        clear_session_vars(tokens)

    assert len(subs) == 2
    source_sub = _sub_by_platform(subs, "photon")
    assert source_sub["delivery_mode"] == "notify+wake"
    assert source_sub["user_id"] == "explicit-user"
    assert source_sub["notifier_profile"] == "operator"
    assert source_sub["delivery_metadata"] == {"explicit": "kept"}
    assert _sub_by_platform(subs, "telegram")["delivery_mode"] == "notify"

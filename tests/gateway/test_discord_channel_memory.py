import json
from pathlib import Path

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture()
def hermes_home(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        yield tmp_path
    finally:
        reset_hermes_home_override(token)


def discord_source(chat_id="chan-a", *, thread_id=None, parent_chat_id=None, chat_name="system-optimization"):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=chat_id,
        chat_name=chat_name,
        chat_type="thread" if thread_id else "channel",
        user_id="user-1",
        user_name="Pete",
        thread_id=thread_id,
        chat_topic="Hermes system work",
        guild_id="guild-1",
        parent_chat_id=parent_chat_id,
        message_id="msg-1",
    )


def test_discord_channel_memory_injects_only_matching_channel(hermes_home):
    from gateway.discord_channel_memory import DiscordChannelMemoryStore

    store = DiscordChannelMemoryStore()
    source_a = discord_source("chan-a")
    source_b = discord_source("chan-b", chat_name="different")

    store.update_after_turn(
        source_a,
        user_text="Decision: use hybrid channel memory for this channel.",
        assistant_text="Implemented: compact JSON state and Obsidian mirror.",
        session_id="session-a",
        message_id="msg-a",
    )

    brief_a = store.build_injection_prompt(source_a)
    brief_b = store.build_injection_prompt(source_b)

    assert "## Discord Channel Operating Memory" in brief_a
    assert "hybrid channel memory" in brief_a
    assert "session-a" in brief_a
    assert brief_b == ""


def test_discord_channel_memory_scopes_threads_separately_from_parent_channel(hermes_home):
    from gateway.discord_channel_memory import DiscordChannelMemoryStore

    store = DiscordChannelMemoryStore()
    parent = discord_source("parent-chan", chat_name="parent")
    thread = discord_source(
        "thread-1",
        thread_id="thread-1",
        parent_chat_id="parent-chan",
        chat_name="implementation-thread",
    )

    store.update_after_turn(
        parent,
        user_text="Decision: parent channel keeps general operations.",
        assistant_text="Saved parent brief.",
        session_id="session-parent",
        message_id="msg-parent",
    )
    store.update_after_turn(
        thread,
        user_text="Decision: thread tracks rollout details.",
        assistant_text="Saved thread brief.",
        session_id="session-thread",
        message_id="msg-thread",
    )

    parent_brief = store.build_injection_prompt(parent)
    thread_brief = store.build_injection_prompt(thread)

    assert "parent channel keeps general operations" in parent_brief
    assert "thread tracks rollout details" not in parent_brief
    assert "thread tracks rollout details" in thread_brief
    assert "parent channel keeps general operations" not in thread_brief


def test_discord_channel_memory_caps_prompt_and_writes_obsidian_mirror(hermes_home):
    from gateway.discord_channel_memory import DiscordChannelMemoryStore

    source = discord_source("chan-a")
    store = DiscordChannelMemoryStore(max_prompt_chars=700, obsidian_root=hermes_home / "vault" / "09 Hermes" / "Discord Channel Context")

    for idx in range(30):
        store.update_after_turn(
            source,
            user_text=f"Decision: durable item {idx} " + ("x" * 80),
            assistant_text="Verified.",
            session_id=f"session-{idx}",
            message_id=f"msg-{idx}",
        )

    brief = store.build_injection_prompt(source)
    assert len(brief) <= 700
    assert "Discord Channel Operating Memory" in brief
    assert "truncated to fit" in brief

    index_path = hermes_home / "state" / "discord_channel_memory" / "index.json"
    index = json.loads(index_path.read_text())
    assert len(index["channels"]) == 1

    record_path = Path(index["channels"][0]["state_path"])
    record = json.loads(record_path.read_text())
    mirror_path = Path(record["obsidian_note_path"])
    assert mirror_path.exists()
    mirror = mirror_path.read_text()
    assert "# Discord Channel Context" in mirror
    assert "guild-1" in mirror

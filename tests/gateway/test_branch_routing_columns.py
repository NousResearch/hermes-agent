"""Regression test for /branch losing gateway routing columns (#NNNNN).

``_handle_branch_command`` (gateway/slash_commands.py) creates the branched
child session via ``create_session()`` WITHOUT chat_id/chat_type/thread_id —
identical in shape to the compression-rotation bug fixed in
agent/conversation_compression.py. The routing columns are only written
later, when ``switch_session()`` calls ``_record_gateway_session_peer()``
after the branch's session_id is live and its transcript has been copied
message-by-message.

A crash/kill landing between create_session() and switch_session() (most
plausibly mid-history-copy on a long conversation, since each
append_message call is independently try/excepted and best-effort) leaves
the branched session permanently unroutable: NULL chat_id/thread_id can
never be found by find_latest_gateway_session_for_peer, and the /resume
IDOR guard (which requires the row's chat_id/thread_id to match the
caller's) can never authorize a manual recovery either.

This test drives the REAL _handle_branch_command against a REAL SessionStore
+ SessionDB (SQLite in a tmp_path, no mocks on the DB/session-store layer)
and asserts the branched child's routing columns are present in state.db
immediately after create_session() returns — before switch_session() ever
runs — closing the gap the same way the compression fix does.
"""

from __future__ import annotations

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, SessionStore, build_session_key
from hermes_state import AsyncSessionDB, SessionDB


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Real SessionStore backed by a real SessionDB (SQLite in tmp_path)."""
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    config = GatewayConfig()
    return SessionStore(sessions_dir=tmp_path, config=config)


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="170829464",
        chat_id="170829464",
        chat_type="dm",
        thread_id="544520",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_branch_runner(store: SessionStore):
    """Minimal GatewayRunner stub wired to a REAL session_store/session_db."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._agent_cache_lock = None
    runner.session_store = store
    runner._session_db = AsyncSessionDB(store._db)
    runner._pending_skills_reload_notes = {}
    return runner


class TestBranchRoutingColumns:
    @pytest.mark.asyncio
    async def test_branched_session_has_routing_columns_before_switch(self, store):
        """Simulates a crash/kill landing between create_session() and
        switch_session() — the exact gap where the branched child's
        routing columns are missing. switch_session() runs unconditionally
        at the end of _handle_branch_command and backfills chat_id/
        chat_type/thread_id via _record_gateway_session_peer(), so checking
        DB state AFTER the function returns successfully will always look
        fine. The real bug only bites if the process dies mid-function,
        before switch_session() gets a chance to run — so we patch
        switch_session to simulate exactly that crash point, then inspect
        the child row switch_session would otherwise have fixed up.
        """
        source = _make_source()
        parent_entry = store.get_or_create_session(source)
        store._db.append_message(parent_entry.session_id, role="user", content="hello")
        store._db.append_message(parent_entry.session_id, role="assistant", content="world")

        runner = _make_branch_runner(store)

        captured_new_session_id = {}
        real_switch_session = store.switch_session

        def _crash_before_switch(session_key, target_session_id):
            # Simulate the process dying right here — before routing gets
            # backfilled — by capturing the id and raising instead of
            # forwarding to the real switch_session().
            captured_new_session_id["id"] = target_session_id
            raise RuntimeError("simulated crash before switch_session")

        import unittest.mock as mock

        with mock.patch.object(store, "switch_session", side_effect=_crash_before_switch):
            with pytest.raises(RuntimeError, match="simulated crash"):
                await runner._handle_branch_command(_make_event("/branch"))

        new_session_id = captured_new_session_id["id"]
        assert new_session_id != parent_entry.session_id

        row = store._db.get_session(new_session_id)
        assert row is not None, "branched child row must exist in state.db"
        # THIS is the bug: without the fix, these are all None, because
        # create_session() at branch time never received them — only the
        # (now-crashed) switch_session() call would have backfilled them.
        assert row["chat_id"] == "170829464", (
            "branched session lost chat_id — unroutable if a crash lands "
            "between create_session() and switch_session()"
        )
        assert row["chat_type"] == "dm"
        assert row["thread_id"] == "544520"
        # user_id and session_key are also required for the fallback lookup
        # path (hermes_state.py:1994-2009) when session_key-based lookup fails
        assert row["user_id"] == "170829464", (
            "branched session lost user_id — fallback peer-tuple lookup "
            "in find_latest_gateway_session_for_peer can never match"
        )
        assert row["session_key"] is not None, (
            "branched session lost session_key — primary lookup path fails"
        )
        # origin_json completes the identity (#82633 reset-path pattern):
        # consumers reading routing/presentation data from state.db
        # (mcp_serve, mirror, channel directory) need the full origin on
        # the branch row without waiting for any backfill.
        assert row["origin_json"], (
            "branched session lost origin_json — state.db consumers see an "
            "identity-less branch row until a peer refresh backfills it"
        )
        import json as _json

        origin = _json.loads(row["origin_json"])
        assert origin.get("chat_id") == "170829464"
        assert origin.get("thread_id") == "544520"

        _ = real_switch_session  # silence unused



@pytest.mark.asyncio
@pytest.mark.parametrize("command,title,in_thread", [
    ("discord-branch", "Alternate path", False),
    ("branch-thread", "", False),
    ("thread-branch", "Sibling path", True),
])
async def test_discord_branch_dispatch_keeps_parent_and_routes_child(store, monkeypatch, command, title, in_thread):
    """#116433: aliases fork history into a distinct, durable thread lane."""
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import discord
    from plugins.platforms.discord.adapter import DiscordAdapter
    from hermes_cli.commands import resolve_command, should_bypass_active_session
    from gateway.session_identity import replace_source

    source = SessionSource(platform=Platform.DISCORD, user_id="42", chat_id="200" if in_thread else "100",
                           thread_id="200" if in_thread else None, chat_type="thread" if in_thread else "channel",
                           parent_chat_id="100" if in_thread else None, guild_id="10")
    parent = store.get_or_create_session(source)
    store._db.append_message(parent.session_id, role="user", content="Explore this")
    store._db.append_message(parent.session_id, role="assistant", content="Original answer")
    runner = _make_branch_runner(store)
    thread = SimpleNamespace(id=300, name=title or "branch")
    create_thread = AsyncMock(return_value=thread)
    class TextChannel(SimpleNamespace):
        pass

    monkeypatch.setattr(discord, "TextChannel", TextChannel)
    monkeypatch.setattr(discord, "DMChannel", type("DMChannel", (), {}))
    channel = TextChannel(id=100, parent=None, create_thread=create_thread)
    if command == "branch-thread":
        create_thread.side_effect = RuntimeError("Direct creation denied")
        channel.send = AsyncMock(return_value=SimpleNamespace(create_thread=AsyncMock(return_value=thread)))
    adapter = object.__new__(DiscordAdapter)
    adapter._client = SimpleNamespace(get_channel=lambda channel_id: (
        SimpleNamespace(parent=channel) if in_thread else channel))
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda src: adapter)
    event = MessageEvent(text=f"/{command} {title}".strip(), source=source, message_id="m1")
    definition = resolve_command(event.get_command())
    assert definition is not None and definition.gateway_only
    assert should_bypass_active_session(command)
    busy = await runner._dispatch_busy_slash_command(event, definition, parent.session_key, source)
    assert "can't run" in busy
    create_thread.assert_not_awaited()

    handled, reply = await runner._hm_dispatch_canonical_command(event, source, parent.session_key, definition.name)
    assert handled and "<#300>" in reply
    create_thread.assert_awaited_once()
    assert create_thread.call_args.kwargs["type"] == discord.ChannelType.public_thread
    if title:
        assert create_thread.call_args.kwargs["name"] == title
    else:
        assert create_thread.call_args.kwargs["name"]
    destination = replace_source(source, chat_id="300", thread_id="300", chat_type="thread", parent_chat_id="100")
    child = store.get_or_create_session(destination)
    assert child.session_id != parent.session_id
    assert store.get_or_create_session(source).session_id == parent.session_id
    row = store._db.get_session(child.session_id)
    assert row["parent_session_id"] == parent.session_id
    assert row["chat_id"] == row["thread_id"] == "300"
    assert row["session_key"] == runner._session_key_for_source(destination)
    assert json.loads(row["origin_json"])["thread_id"] == "300"
    assert store._db.get_session(parent.session_id)["ended_at"] is None
    assert [(m["role"], m["content"]) for m in store.load_transcript(child.session_id)] == [
        ("user", "Explore this"), ("assistant", "Original answer")]
    # Fresh routing-store load proves subsequent thread messages recover the same child.
    restored = SessionStore(sessions_dir=store.sessions_dir, config=GatewayConfig())
    assert restored.get_or_create_session(destination).session_id == child.session_id


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["platform", "unsupported", "permission", "exception", "empty"])
async def test_discord_branch_failure_leaves_parent_untouched(store, monkeypatch, failure):
    """Unsupported contexts and failed thread creation must never fork/switch the parent."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    source = SessionSource(platform=Platform.TELEGRAM if failure == "platform" else Platform.DISCORD,
                           user_id="42", chat_id="100", chat_type="channel")
    parent = store.get_or_create_session(source)
    if failure != "empty":
        store._db.append_message(parent.session_id, role="user", content="Keep me")
    runner = _make_branch_runner(store)
    create_thread = AsyncMock(return_value={"error": "Missing thread permissions"})
    if failure == "exception":
        create_thread.side_effect = RuntimeError("transport unavailable")
    adapter = SimpleNamespace() if failure == "unsupported" else SimpleNamespace(create_thread=create_thread)
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda src: adapter)
    before = [entry.session_id for entry in store.list_sessions()]
    reply = await runner._handle_discord_branch_command(MessageEvent(text="/discord-branch", source=source))
    assert reply
    assert store.get_or_create_session(source).session_id == parent.session_id
    assert [entry.session_id for entry in store.list_sessions()] == before
    assert store._db.get_session(parent.session_id)["ended_at"] is None
    if failure in {"platform", "unsupported", "empty"}:
        create_thread.assert_not_awaited()
    else:
        create_thread.assert_awaited_once()

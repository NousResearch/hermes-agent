"""Tests for the Discord /branch → thread and /merge commands (gateway path).

Covers:
- /branch on Discord in a CHANNEL spawns a thread, binds the BRANCH session to
  the thread's key, and leaves the parent channel's key untouched.
- /branch on Discord INSIDE a thread spawns a SIBLING thread under the same
  parent channel.
- /branch falls back to the classic in-place switch when thread creation fails
  (adapter returns None) and for non-Discord platforms.
- /merge folds a summary into the PARENT session's transcript as one labeled
  user-role message, posts a note to the parent channel, and archives the thread.
- /merge no-op guards: non-Discord, not-a-thread, not-a-branch, empty branch,
  and summary-unavailable.
"""

import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from hermes_state import AsyncSessionDB, SessionDB


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def session_db(tmp_path):
    os.environ["HERMES_HOME"] = str(tmp_path / ".hermes")
    os.makedirs(tmp_path / ".hermes", exist_ok=True)
    db = SessionDB(db_path=tmp_path / ".hermes" / "test_sessions.db")
    yield db
    db.close()


def _entry(session_key, session_id, source):
    return SessionEntry(
        session_key=session_key,
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=source.platform,
        chat_type=source.chat_type,
    )


def _make_runner(session_db, adapter=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter} if adapter is not None else {}
    runner.config = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._agent_cache_lock = None
    runner._session_db = AsyncSessionDB(session_db)

    # Real-ish session store backed by a MagicMock we control per-test.
    runner.session_store = MagicMock()

    # Neutralize side-effect helpers we don't assert on.
    runner._clear_session_boundary_security_state = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner._release_running_agent_state = MagicMock()
    runner._session_key_for_source = lambda src: build_session_key(src)
    # Owner guard + origin resolution — permissive by default; tests override.
    runner._resume_target_allowed = AsyncMock(return_value=True)
    runner._gateway_session_origin_for_id = lambda sid: None
    runner._resume_row_visible = AsyncMock(return_value=True)
    return runner


def _discord_source(chat_type="group", chat_id="parent_chan", thread_id=None,
                    parent_chat_id=None):
    return SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id=chat_id,
        user_name="tester",
        chat_type=chat_type,
        thread_id=thread_id,
        parent_chat_id=parent_chat_id,
    )


def _event(text, source):
    return MessageEvent(text=text, source=source, message_id="m1")


def _seed_session(db, session_id, title="Work", parent=None, model_config=None):
    db.create_session(
        session_id=session_id,
        source="discord",
        model="anthropic/claude-sonnet-4.6",
        model_config=model_config,
        parent_session_id=parent,
    )
    if title:
        db.set_session_title(session_id, title)


# --------------------------------------------------------------------------- #
# /branch — Discord thread spawn
# --------------------------------------------------------------------------- #

class TestBranchDiscordThread:

    @pytest.mark.asyncio
    async def test_branch_in_channel_spawns_thread_and_binds_thread_key(self, session_db):
        adapter = MagicMock()
        adapter.create_handoff_thread = AsyncMock(return_value="thread999")
        adapter.send = AsyncMock()
        runner = _make_runner(session_db, adapter)

        source = _discord_source(chat_type="group", chat_id="parent_chan")
        parent_key = build_session_key(source)
        current = _entry(parent_key, "parent_sess", source)
        _seed_session(session_db, "parent_sess", title="Parent Work")

        runner.session_store.get_or_create_session.return_value = current
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
        # Whatever key switch_session is called with, return a valid entry.
        runner.session_store.switch_session.return_value = _entry(
            "any", "branch_sess", source
        )

        result = await runner._handle_branch_command(_event("/branch idea", source))

        # A thread was created under the PARENT channel.
        adapter.create_handoff_thread.assert_awaited_once()
        args = adapter.create_handoff_thread.await_args.args
        assert args[0] == "parent_chan"  # parent channel id

        # switch_session was called with the THREAD's key, never the parent's.
        switch_keys = [c.args[0] for c in runner.session_store.switch_session.call_args_list]
        assert len(switch_keys) == 1
        thread_key = switch_keys[0]
        assert "thread999" in thread_key
        assert thread_key != parent_key  # parent channel key untouched

        # Confirmation mentions the new thread.
        assert "thread999" in result or "<#thread999>" in result

    @pytest.mark.asyncio
    async def test_branch_stamps_branch_point_len(self, session_db):
        """The branch session records how many messages it inherited from the
        parent, so /merge can later summarize only the post-branch delta."""
        adapter = MagicMock()
        adapter.create_handoff_thread = AsyncMock(return_value="thrX")
        adapter.send = AsyncMock()
        runner = _make_runner(session_db, adapter)

        source = _discord_source(chat_type="group", chat_id="parent_chan")
        _seed_session(session_db, "parent_sess", title="Parent Work")
        runner.session_store.get_or_create_session.return_value = _entry(
            build_session_key(source), "parent_sess", source
        )
        # Parent has 3 inherited messages at branch time.
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        runner.session_store.switch_session.return_value = _entry("any", "b", source)

        await runner._handle_branch_command(_event("/branch idea", source))

        # The new branch session row carries _branch_point_len == 3.
        import json as _json, sqlite3
        conn = sqlite3.connect(str(session_db.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT model_config FROM sessions WHERE parent_session_id = ?",
            ("parent_sess",),
        ).fetchall()
        conn.close()
        assert rows, "no branch child session row found"
        mc = _json.loads(rows[0]["model_config"] or "{}")
        assert mc.get("_branched_from") == "parent_sess"
        assert mc.get("_branch_point_len") == 3

    @pytest.mark.asyncio
    async def test_native_slash_branch_in_thread_spawns_sibling_e2e(self, session_db):
        """END-TO-END: a NATIVE Discord slash /branch inside a thread must build
        a source WITH parent_chat_id (via the real adapter _build_slash_event)
        and spawn a SIBLING thread under the parent channel — not fall back to a
        classic in-place branch. This is the exact path that regressed (#230):
        the native-slash source-builder dropped parent_chat_id.
        """
        import sys
        discord_mod = sys.modules.get("discord")
        # Build the REAL adapter source-builder path.
        from types import SimpleNamespace
        from plugins.platforms.discord import adapter as adapter_mod

        Adapter = getattr(adapter_mod, "DiscordPlatformAdapter", None) or getattr(adapter_mod, "DiscordAdapter")
        a = object.__new__(Adapter)
        a._get_effective_topic = lambda ch, is_thread=False: None
        a._resolve_channel_prompt = lambda cid, pid=None: None
        # capture what build_source produces, but return a real SessionSource
        captured = {}
        def _bs(**kw):
            captured.update(kw)
            return SessionSource(
                platform=Platform.DISCORD,
                chat_id=kw.get("chat_id"),
                chat_type=kw.get("chat_type", "group"),
                user_id=kw.get("user_id"),
                user_name=kw.get("user_name"),
                thread_id=kw.get("session_id"),
                parent_chat_id=kw.get("parent_chat_id"),
            )
        a.build_source = _bs

        # A thread interaction: patch discord.Thread/DMChannel so isinstance works.
        class StubThread: pass
        class StubDM: pass
        orig_t, orig_dm = discord_mod.Thread, discord_mod.DMChannel
        discord_mod.Thread, discord_mod.DMChannel = StubThread, StubDM
        adapter_mod.discord.Thread, adapter_mod.discord.DMChannel = StubThread, StubDM
        try:
            tc = StubThread()
            tc.id = 555
            tc.name = "existing-thread"
            tc.guild = SimpleNamespace(name="Daemonarchy")
            tc.parent = None
            tc.parent_id = 700  # hosting channel
            inter = SimpleNamespace(channel_id=555, channel=tc,
                                    user=SimpleNamespace(id=1, display_name="Ace"))
            event = a._build_slash_event(inter, "/branch")
        finally:
            discord_mod.Thread, discord_mod.DMChannel = orig_t, orig_dm
            adapter_mod.discord.Thread, adapter_mod.discord.DMChannel = orig_t, orig_dm

        # The native-slash source now carries parent_chat_id — the #230 fix.
        assert event.source.parent_chat_id == "700"
        assert event.source.chat_type == "thread"

        # Feed that real source into the branch handler; it must spawn a SIBLING
        # thread under the parent channel (700), not fall back to in-place.
        branch_adapter = MagicMock()
        branch_adapter.create_handoff_thread = AsyncMock(return_value="sibling888")
        branch_adapter.send = AsyncMock()
        runner = _make_runner(session_db, branch_adapter)
        _seed_session(session_db, "src_sess", title="In Thread")
        runner.session_store.get_or_create_session.return_value = _entry(
            build_session_key(event.source), "src_sess", event.source
        )
        runner.session_store.load_transcript.return_value = [{"role": "user", "content": "x"}]
        runner.session_store.switch_session.return_value = _entry("k", "b", event.source)

        result = await runner._handle_branch_command(event)

        branch_adapter.create_handoff_thread.assert_awaited_once()
        assert branch_adapter.create_handoff_thread.await_args.args[0] == "700"  # sibling under parent
        assert "sibling888" in result

    @pytest.mark.asyncio
    async def test_branch_in_thread_spawns_sibling_under_parent(self, session_db):
        adapter = MagicMock()
        adapter.create_handoff_thread = AsyncMock(return_value="sibling777")
        adapter.send = AsyncMock()
        runner = _make_runner(session_db, adapter)

        # We're inside an existing thread; parent_chat_id is the hosting channel.
        source = _discord_source(
            chat_type="thread", chat_id="existing_thread",
            thread_id="existing_thread", parent_chat_id="host_chan",
        )
        current = _entry(build_session_key(source), "cur_sess", source)
        _seed_session(session_db, "cur_sess", title="Thread Work")

        runner.session_store.get_or_create_session.return_value = current
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "hi"},
        ]
        runner.session_store.switch_session.return_value = _entry("any", "b", source)

        await runner._handle_branch_command(_event("/branch", source))

        # Sibling thread is created under the PARENT channel, not the thread.
        args = adapter.create_handoff_thread.await_args.args
        assert args[0] == "host_chan"

    @pytest.mark.asyncio
    async def test_branch_falls_back_when_thread_creation_fails(self, session_db):
        adapter = MagicMock()
        adapter.create_handoff_thread = AsyncMock(return_value=None)  # creation failed
        adapter.send = AsyncMock()
        runner = _make_runner(session_db, adapter)

        source = _discord_source(chat_type="group", chat_id="parent_chan")
        parent_key = build_session_key(source)
        current = _entry(parent_key, "parent_sess", source)
        _seed_session(session_db, "parent_sess", title="Parent Work")

        runner.session_store.get_or_create_session.return_value = current
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "hi"},
        ]
        runner.session_store.switch_session.return_value = _entry(
            parent_key, "branch_sess", source
        )

        result = await runner._handle_branch_command(_event("/branch", source))

        # Classic in-place: switch_session called with the CURRENT (parent) key.
        switch_keys = [c.args[0] for c in runner.session_store.switch_session.call_args_list]
        assert parent_key in switch_keys
        # Classic branch confirmation, not the thread one.
        assert "thread" not in result.lower()

    @pytest.mark.asyncio
    async def test_branch_non_discord_uses_classic_path(self, session_db):
        runner = _make_runner(session_db, adapter=None)  # no discord adapter

        source = SessionSource(
            platform=Platform.TELEGRAM, user_id="u", chat_id="c",
            user_name="t", chat_type="dm",
        )
        key = build_session_key(source)
        current = _entry(key, "parent_sess", source)
        _seed_session(session_db, "parent_sess", title="TG Work")

        runner.session_store.get_or_create_session.return_value = current
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "hi"},
        ]
        runner.session_store.switch_session.return_value = _entry(key, "branch_sess", source)

        result = await runner._handle_branch_command(_event("/branch", source))
        switch_keys = [c.args[0] for c in runner.session_store.switch_session.call_args_list]
        assert key in switch_keys
        assert "thread" not in result.lower()


# --------------------------------------------------------------------------- #
# /merge
# --------------------------------------------------------------------------- #

class TestMergeCommand:

    def _runner_with_summary(self, session_db, adapter=None, summary="SUMMARY BODY"):
        runner = _make_runner(session_db, adapter)
        fake = MagicMock()
        fake._generate_summary = MagicMock(return_value=summary)
        runner._build_merge_summarizer = lambda: fake
        return runner

    # --- Thread form: /merge (no arg) inside a branched Discord thread --------

    @pytest.mark.asyncio
    async def test_merge_thread_folds_into_parent_and_archives(self, session_db):
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter.archive_thread = AsyncMock(return_value=True)
        runner = self._runner_with_summary(session_db, adapter)

        _seed_session(session_db, "parent_sess", title="Parent Work")
        _seed_session(session_db, "branch_sess", title="Explore X", parent="parent_sess")

        source = _discord_source(
            chat_type="thread", chat_id="thread123",
            thread_id="thread123", parent_chat_id="parent_chan",
        )
        current = _entry(build_session_key(source), "branch_sess", source)
        runner.session_store.get_or_create_session.return_value = current
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "explore this"},
            {"role": "assistant", "content": "explored"},
        ]
        parent_src = _discord_source(chat_type="group", chat_id="parent_chan")
        runner.session_store.lookup_by_session_id.return_value = _entry(
            build_session_key(parent_src), "parent_sess", parent_src
        )
        # Target (parent) session's live origin — the note must land HERE (where
        # the fold went), not in source.parent_chat_id.
        runner._gateway_session_origin_for_id = lambda sid: parent_src

        result = await runner._handle_merge_command(_event("/merge", source))

        # Exactly one labeled thread-fold into the PARENT transcript.
        parent_msgs = session_db.get_messages_as_conversation("parent_sess")
        folds = [m for m in parent_msgs
                 if m.get("role") == "user"
                 and "BRANCHED THREAD MERGED" in str(m.get("content", ""))]
        assert len(folds) == 1
        assert "SUMMARY BODY" in folds[0]["content"]

        # Note posted to the TARGET SESSION's origin (parent_chan here); thread
        # archived; confirm says so.
        adapter.send.assert_awaited()
        assert adapter.send.await_args.args[0] == "parent_chan"
        adapter.archive_thread.assert_awaited_once()
        assert "archived" in result.lower()
        runner._evict_cached_agent.assert_called()

    @pytest.mark.asyncio
    async def test_merge_thread_summarizes_only_delta_after_branch_point(self, session_db):
        """Thread form summarizes ONLY turns after the branch point, not the
        whole inherited copy of the parent."""
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter.archive_thread = AsyncMock(return_value=True)

        runner = _make_runner(session_db, adapter)
        seen = {}
        fake = MagicMock()
        def _cap_summary(history, focus=None):
            seen["history"] = history
            return "DELTA SUMMARY"
        fake._generate_summary = _cap_summary
        runner._build_merge_summarizer = lambda: fake

        _seed_session(session_db, "parent_sess", title="Parent")
        # Branch inherited 2 messages from the parent (branch point = 2).
        _seed_session(session_db, "branch_sess", title="Explore",
                      parent="parent_sess",
                      model_config={"_branched_from": "parent_sess", "_branch_point_len": 2})

        source = _discord_source(chat_type="thread", chat_id="t9",
                                 thread_id="t9", parent_chat_id="chan")
        current = _entry(build_session_key(source), "branch_sess", source)
        runner.session_store.get_or_create_session.return_value = current
        # Full branch transcript: 2 inherited + 2 new exploration turns.
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "INHERITED q"},
            {"role": "assistant", "content": "INHERITED a"},
            {"role": "user", "content": "NEW exploration q"},
            {"role": "assistant", "content": "NEW exploration a"},
        ]
        runner.session_store.lookup_by_session_id.return_value = None

        await runner._handle_merge_command(_event("/merge", source))

        # The summarizer saw ONLY the 2 post-branch turns.
        summed = seen.get("history") or []
        contents = " ".join(str(m.get("content", "")) for m in summed)
        assert len(summed) == 2
        assert "NEW exploration" in contents
        assert "INHERITED" not in contents

    @pytest.mark.asyncio
    async def test_merge_thread_no_new_turns_does_not_refold(self, session_db):
        """A branch with ZERO new turns (branch_point_len == len) must return
        no_new_turns and fold NOTHING — not re-summarize the inherited copy."""
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter.archive_thread = AsyncMock(return_value=True)
        runner = self._runner_with_summary(session_db, adapter)

        _seed_session(session_db, "parent_sess", title="Parent")
        _seed_session(session_db, "branch_sess", title="Explore",
                      parent="parent_sess",
                      model_config={"_branched_from": "parent_sess", "_branch_point_len": 2})
        source = _discord_source(chat_type="thread", chat_id="t0",
                                 thread_id="t0", parent_chat_id="chan")
        current = _entry(build_session_key(source), "branch_sess", source)
        runner.session_store.get_or_create_session.return_value = current
        # Exactly the 2 inherited turns, nothing new.
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "INHERITED q"},
            {"role": "assistant", "content": "INHERITED a"},
        ]
        runner.session_store.lookup_by_session_id.return_value = None

        result = await runner._handle_merge_command(_event("/merge", source))

        assert "nothing new" in result.lower() or "no turns" in result.lower()
        parent_msgs = session_db.get_messages_as_conversation("parent_sess")
        assert not any("BRANCHED THREAD MERGED" in str(m.get("content", ""))
                       for m in parent_msgs)
        adapter.archive_thread.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_merge_thread_uses_branched_from_marker(self, session_db):
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter.archive_thread = AsyncMock(return_value=True)
        runner = self._runner_with_summary(session_db, adapter)

        _seed_session(session_db, "parent_sess", title="Parent Work")
        _seed_session(session_db, "branch_sess", title="Explore",
                      model_config={"_branched_from": "parent_sess"})

        source = _discord_source(
            chat_type="thread", chat_id="t2", thread_id="t2",
            parent_chat_id="parent_chan",
        )
        current = _entry(build_session_key(source), "branch_sess", source)
        runner.session_store.get_or_create_session.return_value = current
        runner.session_store.load_transcript.return_value = [{"role": "user", "content": "x"}]
        runner.session_store.lookup_by_session_id.return_value = None

        await runner._handle_merge_command(_event("/merge", source))
        parent_msgs = session_db.get_messages_as_conversation("parent_sess")
        assert any("BRANCHED THREAD MERGED" in str(m.get("content", ""))
                   for m in parent_msgs)

    # --- Named form: /merge <name>, any platform -----------------------------

    @pytest.mark.asyncio
    async def test_merge_named_folds_into_target_any_platform(self, session_db):
        """/merge <title> on Telegram folds this session's summary into the named one."""
        runner = self._runner_with_summary(session_db, adapter=None)

        _seed_session(session_db, "target_sess", title="Target Convo")
        _seed_session(session_db, "current_sess", title="Here")

        source = SessionSource(
            platform=Platform.TELEGRAM, user_id="u", chat_id="c",
            user_name="t", chat_type="dm",
        )
        current = _entry(build_session_key(source), "current_sess", source)
        runner.session_store.get_or_create_session.return_value = current
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "some work here"},
            {"role": "assistant", "content": "done"},
        ]
        runner.session_store.lookup_by_session_id.return_value = None

        result = await runner._handle_merge_command(_event("/merge Target Convo", source))

        target_msgs = session_db.get_messages_as_conversation("target_sess")
        folds = [m for m in target_msgs
                 if m.get("role") == "user"
                 and "MERGED SESSION" in str(m.get("content", ""))]
        assert len(folds) == 1
        assert "SUMMARY BODY" in folds[0]["content"]
        assert "merged into" in result.lower()
        # The SOURCE session is summarized (read-only) but gets a small marker
        # appended so a resumed branch knows it was already merged — it must NOT
        # receive the full fold, only the [MERGED — …] breadcrumb.
        cur_msgs = session_db.get_messages_as_conversation("current_sess")
        assert not any("MERGED SESSION" in str(m.get("content", "")) for m in cur_msgs)
        markers = [m for m in cur_msgs if "[MERGED —" in str(m.get("content", ""))]
        assert len(markers) == 1
        assert "Target Convo" in markers[0]["content"]
        # Named form (no branch/delta): the marker must use the whole-session
        # wording, NOT the thread-specific "this thread's new exploration".
        assert "new exploration" not in markers[0]["content"]
        assert "a summary of this session" in markers[0]["content"]

    @pytest.mark.asyncio
    async def test_merge_note_is_terse_turn_count_not_summary_dump(self, session_db):
        """The visible 'merged in' note states the turn count + record path — it
        does NOT dump the full summary into the channel (Ace, 2026-07-08)."""
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter.archive_thread = AsyncMock(return_value=True)
        runner = self._runner_with_summary(
            session_db, adapter, summary="THIS LONG SUMMARY SHOULD NOT APPEAR IN THE NOTE")

        _seed_session(session_db, "parent_sess", title="Parent")
        _seed_session(session_db, "branch_sess", title="Explore",
                      parent="parent_sess",
                      model_config={"_branched_from": "parent_sess", "_branch_point_len": 1})
        source = _discord_source(chat_type="thread", chat_id="tN",
                                 thread_id="tN", parent_chat_id="host")
        current = _entry(build_session_key(source), "branch_sess", source)
        runner.session_store.get_or_create_session.return_value = current
        # 1 inherited + 2 new user turns after the branch point.
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "inherited"},
            {"role": "user", "content": "new q1"},
            {"role": "assistant", "content": "new a1"},
            {"role": "user", "content": "new q2"},
        ]
        parent_src = _discord_source(chat_type="group", chat_id="parent_origin")
        runner.session_store.lookup_by_session_id.return_value = _entry(
            build_session_key(parent_src), "parent_sess", parent_src)
        runner._gateway_session_origin_for_id = lambda sid: parent_src

        await runner._handle_merge_command(_event("/merge", source))

        adapter.send.assert_awaited()
        note = adapter.send.await_args.args[1]
        # Note goes to the target's origin, mentions the 2 new user turns, and
        # does NOT contain the raw summary body.
        assert adapter.send.await_args.args[0] == "parent_origin"
        assert "2 new turn" in note
        assert "THIS LONG SUMMARY" not in note

    @pytest.mark.asyncio
    async def test_merge_named_note_omits_new_turn_wording(self, session_db):
        """Named-form (/merge <name>) folds the WHOLE session, so the visible
        note must NOT claim "N new turn(s)" — that wording is thread/delta
        specific. It uses the _named variant instead (Greptile P2, 2026-07-08)."""
        adapter = MagicMock()
        adapter.send = AsyncMock()
        runner = self._runner_with_summary(session_db, adapter)

        _seed_session(session_db, "target_sess", title="Target Convo")
        _seed_session(session_db, "current_sess", title="Here")
        source = SessionSource(
            platform=Platform.TELEGRAM, user_id="u", chat_id="c",
            user_name="t", chat_type="dm",
        )
        current = _entry(build_session_key(source), "current_sess", source)
        runner.session_store.get_or_create_session.return_value = current
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        # Target lives on a Discord origin (the adapter the runner has wired) so
        # the "Folded in" note is actually posted. The MERGE is still the named
        # form (/merge <name>) — that's what selects the _named wording.
        target_src = _discord_source(chat_type="group", chat_id="target_origin")
        runner.session_store.lookup_by_session_id.return_value = _entry(
            build_session_key(target_src), "target_sess", target_src)
        runner._gateway_session_origin_for_id = lambda sid: target_src

        await runner._handle_merge_command(_event("/merge Target Convo", source))

        adapter.send.assert_awaited()
        # Find the posted merge note (the "Folded in" one).
        notes = [c.args[1] for c in adapter.send.await_args_list
                 if "Folded in" in str(c.args[1])]
        assert notes, "expected a 'Folded in' note to be posted"
        note = notes[0]
        assert "new turn" not in note        # named form: not the delta wording
        assert "Here" in note                # note names the SOURCE session

    @pytest.mark.asyncio
    async def test_merge_named_writes_md_record(self, session_db, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        runner = self._runner_with_summary(session_db, adapter=None)
        _seed_session(session_db, "target_sess", title="Target")
        _seed_session(session_db, "current_sess", title="Source Here")
        source = SessionSource(platform=Platform.TELEGRAM, user_id="u", chat_id="c",
                               user_name="t", chat_type="dm")
        current = _entry(build_session_key(source), "current_sess", source)
        runner.session_store.get_or_create_session.return_value = current
        runner.session_store.load_transcript.return_value = [{"role": "user", "content": "x"}]
        runner.session_store.lookup_by_session_id.return_value = None

        result = await runner._handle_merge_command(_event("/merge Target", source))

        merges_dir = tmp_path / ".hermes" / "merges"
        files = list(merges_dir.glob("*.md"))
        assert len(files) == 1
        body = files[0].read_text()
        assert "SUMMARY BODY" in body and "Target" in body
        assert "full record" in result.lower()

    @pytest.mark.asyncio
    async def test_merge_named_blocked_cross_owner(self, session_db):
        """Owner guard: refuse merging into a session you don't own."""
        runner = self._runner_with_summary(session_db, adapter=None)
        runner._resume_target_allowed = AsyncMock(return_value=False)  # not owner

        _seed_session(session_db, "target_sess", title="Someone Elses")
        _seed_session(session_db, "current_sess", title="Here")
        source = SessionSource(platform=Platform.TELEGRAM, user_id="u", chat_id="c",
                               user_name="t", chat_type="dm")
        current = _entry(build_session_key(source), "current_sess", source)
        runner.session_store.get_or_create_session.return_value = current

        result = await runner._handle_merge_command(_event("/merge Someone Elses", source))
        # Nothing folded; blocked message returned.
        target_msgs = session_db.get_messages_as_conversation("target_sess")
        assert not any("MERGED SESSION" in str(m.get("content", "")) for m in target_msgs)
        assert "own" in result.lower()

    @pytest.mark.asyncio
    async def test_merge_named_not_found(self, session_db):
        runner = self._runner_with_summary(session_db, adapter=None)
        _seed_session(session_db, "current_sess", title="Here")
        source = SessionSource(platform=Platform.TELEGRAM, user_id="u", chat_id="c",
                               user_name="t", chat_type="dm")
        current = _entry(build_session_key(source), "current_sess", source)
        runner.session_store.get_or_create_session.return_value = current
        result = await runner._handle_merge_command(_event("/merge NoSuchName", source))
        assert "no session" in result.lower() or "matching" in result.lower()

    @pytest.mark.asyncio
    async def test_merge_same_session_refused(self, session_db):
        runner = self._runner_with_summary(session_db, adapter=None)
        _seed_session(session_db, "current_sess", title="Here")
        source = SessionSource(platform=Platform.TELEGRAM, user_id="u", chat_id="c",
                               user_name="t", chat_type="dm")
        current = _entry(build_session_key(source), "current_sess", source)
        runner.session_store.get_or_create_session.return_value = current
        # Resolve the name to the SAME id as current.
        result = await runner._handle_merge_command(_event("/merge current_sess", source))
        assert "already in" in result.lower() or "different target" in result.lower()

    # --- No-arg, not a branched thread → list mergeable sessions -------------

    @pytest.mark.asyncio
    async def test_merge_no_arg_lists_mergeable_sessions(self, session_db):
        runner = self._runner_with_summary(session_db, adapter=None)
        _seed_session(session_db, "current_sess", title="Here")
        _seed_session(session_db, "other_a", title="Alpha")
        _seed_session(session_db, "other_b", title="Beta")
        source = SessionSource(platform=Platform.TELEGRAM, user_id="u", chat_id="c",
                               user_name="t", chat_type="dm")
        current = _entry(build_session_key(source), "current_sess", source)
        runner.session_store.get_or_create_session.return_value = current

        result = await runner._handle_merge_command(_event("/merge", source))
        # Lists titled sessions; does NOT fold anything.
        assert "merge into" in result.lower() or "Alpha" in result or "Beta" in result
        assert "MERGED SESSION" not in result

    @pytest.mark.asyncio
    async def test_merge_no_summary_does_not_fold(self, session_db):
        runner = self._runner_with_summary(session_db, adapter=None, summary=None)
        _seed_session(session_db, "target_sess", title="Target")
        _seed_session(session_db, "current_sess", title="Here")
        source = SessionSource(platform=Platform.TELEGRAM, user_id="u", chat_id="c",
                               user_name="t", chat_type="dm")
        current = _entry(build_session_key(source), "current_sess", source)
        runner.session_store.get_or_create_session.return_value = current
        runner.session_store.load_transcript.return_value = [{"role": "user", "content": "x"}]

        result = await runner._handle_merge_command(_event("/merge Target", source))
        target_msgs = session_db.get_messages_as_conversation("target_sess")
        assert not any("MERGED SESSION" in str(m.get("content", "")) for m in target_msgs)
        assert "summary" in result.lower() or "nothing" in result.lower()

    @pytest.mark.asyncio
    async def test_merge_purges_old_records(self, session_db, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        import os as _os, time as _time
        runner = self._runner_with_summary(session_db, adapter=None)
        merges_dir = tmp_path / ".hermes" / "merges"
        merges_dir.mkdir(parents=True)
        # An old record (40 days) + a fresh one.
        old = merges_dir / "old.md"
        old.write_text("stale")
        old_time = _time.time() - 40 * 86400
        _os.utime(old, (old_time, old_time))
        fresh = merges_dir / "fresh.md"
        fresh.write_text("recent")

        _seed_session(session_db, "target_sess", title="Target")
        _seed_session(session_db, "current_sess", title="Here")
        source = SessionSource(platform=Platform.TELEGRAM, user_id="u", chat_id="c",
                               user_name="t", chat_type="dm")
        current = _entry(build_session_key(source), "current_sess", source)
        runner.session_store.get_or_create_session.return_value = current
        runner.session_store.load_transcript.return_value = [{"role": "user", "content": "x"}]
        runner.session_store.lookup_by_session_id.return_value = None

        await runner._handle_merge_command(_event("/merge Target", source))
        assert not old.exists()   # 40-day-old record purged
        assert fresh.exists()     # fresh record kept

    @pytest.mark.asyncio
    async def test_merge_is_idempotent_same_target(self, session_db):
        """A second /merge of the SAME source into the SAME target is a no-op."""
        runner = self._runner_with_summary(session_db, adapter=None)
        _seed_session(session_db, "target_sess", title="Target")
        _seed_session(session_db, "current_sess", title="Here")
        source = SessionSource(platform=Platform.TELEGRAM, user_id="u", chat_id="c",
                               user_name="t", chat_type="dm")
        current = _entry(build_session_key(source), "current_sess", source)
        runner.session_store.get_or_create_session.return_value = current
        runner.session_store.load_transcript.return_value = [{"role": "user", "content": "x"}]
        runner.session_store.lookup_by_session_id.return_value = None

        first = await runner._handle_merge_command(_event("/merge Target", source))
        assert "merged into" in first.lower()
        second = await runner._handle_merge_command(_event("/merge Target", source))
        # Second is refused; only ONE fold in the target.
        assert "already" in second.lower()
        target_msgs = session_db.get_messages_as_conversation("target_sess")
        folds = [m for m in target_msgs
                 if m.get("role") == "user"
                 and "MERGED SESSION" in str(m.get("content", ""))]
        assert len(folds) == 1

    @pytest.mark.asyncio
    async def test_merge_same_source_different_targets_allowed(self, session_db):
        """Merging one source into TWO different targets is allowed (not blocked)."""
        runner = self._runner_with_summary(session_db, adapter=None)
        _seed_session(session_db, "target_a", title="Alpha")
        _seed_session(session_db, "target_b", title="Beta")
        _seed_session(session_db, "current_sess", title="Here")
        source = SessionSource(platform=Platform.TELEGRAM, user_id="u", chat_id="c",
                               user_name="t", chat_type="dm")
        current = _entry(build_session_key(source), "current_sess", source)
        runner.session_store.get_or_create_session.return_value = current
        runner.session_store.load_transcript.return_value = [{"role": "user", "content": "x"}]
        runner.session_store.lookup_by_session_id.return_value = None

        r1 = await runner._handle_merge_command(_event("/merge Alpha", source))
        r2 = await runner._handle_merge_command(_event("/merge Beta", source))
        assert "merged into" in r1.lower() and "merged into" in r2.lower()
        assert any("MERGED SESSION" in str(m.get("content", ""))
                   for m in session_db.get_messages_as_conversation("target_a"))
        assert any("MERGED SESSION" in str(m.get("content", ""))
                   for m in session_db.get_messages_as_conversation("target_b"))


    @pytest.mark.asyncio
    async def test_merge_concurrent_calls_fold_once(self, session_db):
        """Two overlapping /merge calls for the same pair append exactly one fold."""
        import asyncio
        runner = self._runner_with_summary(session_db, adapter=None)
        _seed_session(session_db, "target_sess", title="Target")
        _seed_session(session_db, "current_sess", title="Here")
        source = SessionSource(platform=Platform.TELEGRAM, user_id="u", chat_id="c",
                               user_name="t", chat_type="dm")
        current = _entry(build_session_key(source), "current_sess", source)
        runner.session_store.get_or_create_session.return_value = current
        runner.session_store.load_transcript.return_value = [{"role": "user", "content": "x"}]
        runner.session_store.lookup_by_session_id.return_value = None

        # Fire both concurrently; the per-pair lock must serialize them so only
        # one records+appends and the other sees the marker and no-ops.
        r1, r2 = await asyncio.gather(
            runner._handle_merge_command(_event("/merge Target", source)),
            runner._handle_merge_command(_event("/merge Target", source)),
        )
        outcomes = sorted([r1.lower(), r2.lower()])
        # Exactly one "merged into" and one "already".
        assert any("merged into" in o for o in outcomes)
        assert any("already" in o for o in outcomes)
        folds = [m for m in session_db.get_messages_as_conversation("target_sess")
                 if "MERGED SESSION" in str(m.get("content", ""))]
        assert len(folds) == 1

    @pytest.mark.asyncio
    async def test_merge_refuses_when_ledger_cannot_record(self, session_db):
        """If the idempotency marker can't be durably recorded, no fold is written."""
        runner = self._runner_with_summary(session_db, adapter=None)
        _seed_session(session_db, "target_sess", title="Target")
        _seed_session(session_db, "current_sess", title="Here")
        source = SessionSource(platform=Platform.TELEGRAM, user_id="u", chat_id="c",
                               user_name="t", chat_type="dm")
        current = _entry(build_session_key(source), "current_sess", source)
        runner.session_store.get_or_create_session.return_value = current
        runner.session_store.load_transcript.return_value = [{"role": "user", "content": "x"}]
        runner.session_store.lookup_by_session_id.return_value = None

        # Make the ledger write fail (update_session_meta raises).
        def boom(*a, **k):
            raise RuntimeError("db locked")
        session_db.update_session_meta = boom

        result = await runner._handle_merge_command(_event("/merge Target", source))
        # Fold refused; nothing written to the target.
        assert "record" in result.lower() or "busy" in result.lower()
        folds = [m for m in session_db.get_messages_as_conversation("target_sess")
                 if "MERGED SESSION" in str(m.get("content", ""))]
        assert len(folds) == 0

    @pytest.mark.asyncio
    async def test_merge_ledger_rolls_back_on_append_failure(self, session_db):
        """If the fold append fails, the ledger entry is rolled back so a retry works."""
        runner = self._runner_with_summary(session_db, adapter=None)
        _seed_session(session_db, "target_sess", title="Target")
        _seed_session(session_db, "current_sess", title="Here")
        source = SessionSource(platform=Platform.TELEGRAM, user_id="u", chat_id="c",
                               user_name="t", chat_type="dm")
        current = _entry(build_session_key(source), "current_sess", source)
        runner.session_store.get_or_create_session.return_value = current
        runner.session_store.load_transcript.return_value = [{"role": "user", "content": "x"}]
        runner.session_store.lookup_by_session_id.return_value = None

        # Make the first append raise, then subsequent ones succeed.
        real_append = session_db.append_message
        calls = {"n": 0}
        def flaky_append(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return real_append(*a, **k)
        session_db.append_message = flaky_append

        first = await runner._handle_merge_command(_event("/merge Target", source))
        assert "failed" in first.lower()
        # Ledger must NOT record it (rolled back) — a retry is allowed and succeeds.
        assert not await runner._merge_already_done("current_sess", "target_sess")
        second = await runner._handle_merge_command(_event("/merge Target", source))
        assert "merged into" in second.lower()
        folds = [m for m in session_db.get_messages_as_conversation("target_sess")
                 if "MERGED SESSION" in str(m.get("content", ""))]
        assert len(folds) == 1


class TestMergeCommandDef:

    def test_merge_in_registry_gateway_only(self):
        from hermes_cli.commands import COMMAND_REGISTRY
        merge = next((c for c in COMMAND_REGISTRY if c.name == "merge"), None)
        assert merge is not None
        assert merge.gateway_only is True
        assert merge.category == "Session"
        assert merge.args_hint == "[name]"

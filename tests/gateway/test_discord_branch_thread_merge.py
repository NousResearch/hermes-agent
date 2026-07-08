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

        result = await runner._handle_merge_command(_event("/merge", source))

        # Exactly one labeled thread-fold into the PARENT transcript.
        parent_msgs = session_db.get_messages_as_conversation("parent_sess")
        folds = [m for m in parent_msgs
                 if m.get("role") == "user"
                 and "BRANCHED THREAD MERGED" in str(m.get("content", ""))]
        assert len(folds) == 1
        assert "SUMMARY BODY" in folds[0]["content"]

        # Note posted to the PARENT channel; thread archived; confirm says so.
        adapter.send.assert_awaited()
        assert adapter.send.await_args.args[0] == "parent_chan"
        adapter.archive_thread.assert_awaited_once()
        assert "archived" in result.lower()
        runner._evict_cached_agent.assert_called()

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
        # current session's transcript is NOT mutated (read-only).
        cur_msgs = session_db.get_messages_as_conversation("current_sess")
        assert not any("MERGED SESSION" in str(m.get("content", "")) for m in cur_msgs)

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

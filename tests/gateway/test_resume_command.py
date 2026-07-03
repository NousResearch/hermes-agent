"""Tests for /resume gateway slash command.

Tests the _handle_resume_command handler (switch to a previously-named session)
across gateway messenger platforms.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, build_session_key


def _make_event(text="/resume", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _session_key_for_event(event):
    """Get the session key that build_session_key produces for an event."""
    return build_session_key(event.source)


def _make_runner(session_db=None, current_session_id="current_session_001",
                 event=None):
    """Create a bare GatewayRunner with a mock session_store and optional session_db."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = SimpleNamespace(platforms={})
    runner._voice_mode = {}
    # Gateway holds the async facade; the slash handlers await it.
    if session_db is not None:
        from hermes_state import AsyncSessionDB
        session_db = AsyncSessionDB(session_db)
    runner._session_db = session_db
    runner._running_agents = {}
    runner._is_user_authorized = lambda _source: True

    # Compute the real session key if an event is provided
    session_key = build_session_key(event.source) if event else "agent:main:telegram:dm"

    # Mock session_store that returns a session entry with a known session_id
    mock_session_entry = MagicMock()
    mock_session_entry.session_id = current_session_id
    mock_session_entry.session_key = session_key
    mock_store = MagicMock()
    mock_store.get_or_create_session.return_value = mock_session_entry
    mock_store.load_transcript.return_value = []
    mock_store.switch_session.return_value = mock_session_entry
    runner.session_store = mock_store

    return runner


# ---------------------------------------------------------------------------
# _handle_resume_command
# ---------------------------------------------------------------------------


class TestHandleResumeCommand:
    """Tests for GatewayRunner._handle_resume_command."""

    @pytest.mark.asyncio
    async def test_no_session_db(self):
        """Returns error when session database is unavailable."""
        runner = _make_runner(session_db=None)
        event = _make_event(text="/resume My Project")
        result = await runner._handle_resume_command(event)
        assert "not available" in result.lower()

    @pytest.mark.asyncio
    async def test_list_named_sessions_when_no_arg(self, tmp_path):
        """With no argument, lists recently titled sessions."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_001", "telegram", user_id="12345", chat_id="67890")
        db.create_session("sess_002", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("sess_001", "Research")
        db.set_session_title("sess_002", "Coding")

        event = _make_event(text="/resume")
        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_resume_command(event)
        assert "Research" in result
        assert "Coding" in result
        assert "Named Sessions" in result
        assert "1." in result
        assert "2." in result
        assert "/resume 1" in result
        db.close()

    @pytest.mark.asyncio
    async def test_list_shows_usage_when_no_titled(self, tmp_path):
        """With no arg and no titled sessions, shows instructions."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_001", "telegram", user_id="12345", chat_id="67890")  # No title

        event = _make_event(text="/resume")
        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_resume_command(event)
        assert "No named sessions" in result
        assert "/title" in result
        db.close()

    @pytest.mark.asyncio
    async def test_resume_by_index(self, tmp_path):
        """Numeric argument resumes the indexed titled session from the list."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_001", "telegram", user_id="12345", chat_id="67890")
        db.create_session("sess_002", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("sess_001", "Research")
        db.set_session_title("sess_002", "Coding")
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        event = _make_event(text="/resume 2")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        result = await runner._handle_resume_command(event)

        assert "Resumed" in result
        runner.session_store.switch_session.assert_called_once()
        call_args = runner.session_store.switch_session.call_args
        assert call_args[0][1] == "sess_001"
        db.close()

    @pytest.mark.asyncio
    async def test_resume_index_out_of_range(self, tmp_path):
        """Out-of-range numeric arguments show a helpful error."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_001", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("sess_001", "Research")
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        event = _make_event(text="/resume 9")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        result = await runner._handle_resume_command(event)

        assert "out of range" in result.lower()
        assert "/resume" in result
        runner.session_store.switch_session.assert_not_called()
        db.close()

    @pytest.mark.asyncio
    async def test_resume_by_name(self, tmp_path):
        """Resolves a title and switches to that session."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("old_session_abc", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("old_session_abc", "My Project")
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        event = _make_event(text="/resume My Project")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        result = await runner._handle_resume_command(event)

        assert "Resumed" in result
        assert "My Project" in result
        # Verify switch_session was called with the old session ID
        runner.session_store.switch_session.assert_called_once()
        call_args = runner.session_store.switch_session.call_args
        assert call_args[0][1] == "old_session_abc"
        db.close()

    @pytest.mark.asyncio
    async def test_resume_clears_session_model_overrides(self, tmp_path):
        """Resume must not carry a previous session's /model override into the
        restored conversation, while leaving other chats' overrides intact (#10702)."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("old_session_abc", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("old_session_abc", "My Project")
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        event = _make_event(text="/resume My Project")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        key = _session_key_for_event(event)
        runner._session_model_overrides = {
            key: {"model": "gpt-5", "provider": "openai"},
            "agent:main:telegram:dm:other": {"model": "keep-me"},
        }
        runner._pending_model_notes = {
            key: "[Note: switched to gpt-5]",
            "agent:main:telegram:dm:other": "[Note: keep-me]",
        }

        result = await runner._handle_resume_command(event)

        assert "Resumed" in result
        # The resumed chat's override + pending note are cleared...
        assert key not in runner._session_model_overrides
        assert key not in runner._pending_model_notes
        # ...but an unrelated chat's state is untouched.
        assert runner._session_model_overrides["agent:main:telegram:dm:other"] == {"model": "keep-me"}
        assert runner._pending_model_notes["agent:main:telegram:dm:other"] == "[Note: keep-me]"
        db.close()

    @pytest.mark.asyncio
    async def test_resume_nonexistent_name(self, tmp_path):
        """Returns error for unknown session name."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        event = _make_event(text="/resume Nonexistent Session")
        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_resume_command(event)
        assert "No session found" in result
        db.close()

    @pytest.mark.asyncio
    async def test_resume_already_on_session(self, tmp_path):
        """Returns friendly message when already on the requested session."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("current_session_001", "Active Project")

        event = _make_event(text="/resume Active Project")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        result = await runner._handle_resume_command(event)
        assert "Already on session" in result
        db.close()

    @pytest.mark.asyncio
    async def test_resume_auto_lineage(self, tmp_path):
        """Asking for 'My Project' when 'My Project #2' exists gets the latest."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_v1", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("sess_v1", "My Project")
        db.create_session("sess_v2", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("sess_v2", "My Project #2")
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        event = _make_event(text="/resume My Project")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        result = await runner._handle_resume_command(event)

        assert "Resumed" in result
        # Should resolve to #2 (latest in lineage)
        call_args = runner.session_store.switch_session.call_args
        assert call_args[0][1] == "sess_v2"
        db.close()

    @pytest.mark.asyncio
    async def test_resume_follows_compression_continuation(self, tmp_path):
        """Gateway /resume should reopen the live descendant after compression."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("compressed_root", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("compressed_root", "Compressed Work")
        db.end_session("compressed_root", "compression")
        db.create_session("compressed_child", "telegram", user_id="12345", chat_id="67890", parent_session_id="compressed_root")
        db.append_message("compressed_child", "user", "hello from continuation")
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        event = _make_event(text="/resume Compressed Work")
        runner = _make_runner(
            session_db=db,
            current_session_id="current_session_001",
            event=event,
        )
        runner.session_store.load_transcript.side_effect = (
            lambda session_id: [{"role": "user", "content": "hello from continuation"}]
            if session_id == "compressed_child"
            else []
        )

        result = await runner._handle_resume_command(event)

        assert "Resumed session" in result
        assert "(1 message)" in result
        call_args = runner.session_store.switch_session.call_args
        assert call_args[0][1] == "compressed_child"
        runner.session_store.load_transcript.assert_called_with("compressed_child")
        db.close()

    @pytest.mark.asyncio
    async def test_resume_clears_running_agent(self, tmp_path):
        """Switching sessions clears any cached running agent."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("old_session", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("old_session", "Old Work")
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        event = _make_event(text="/resume Old Work")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        # Simulate a running agent using the real session key
        real_key = _session_key_for_event(event)
        runner._running_agents[real_key] = MagicMock()

        await runner._handle_resume_command(event)

        assert real_key not in runner._running_agents
        db.close()

    @pytest.mark.asyncio
    async def test_resume_evicts_cached_agent(self, tmp_path):
        """Gateway /resume evicts the cached AIAgent so the next message
        rebuilds with the correct session_id end-to-end — mirrors /branch
        and /reset. Without this, the cached agent's memory provider keeps
        writing into the wrong session. See #6672.
        """
        import threading
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("old_session", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("old_session", "Old Work")
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        event = _make_event(text="/resume Old Work")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        # Seed the cache with a fake agent
        real_key = _session_key_for_event(event)
        runner._agent_cache = {real_key: (MagicMock(), object())}
        runner._agent_cache_lock = threading.RLock()

        await runner._handle_resume_command(event)

        assert real_key not in runner._agent_cache
        db.close()

    @pytest.mark.asyncio
    async def test_resume_strips_outer_brackets(self, tmp_path):
        """Users may copy `<session_id>` from the usage hint literally.

        The gateway should strip outer ``<>``, ``[]``, ``""``, and ``''``
        before lookup so ``/resume <abc123>`` works the same as
        ``/resume abc123``.
        """
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("abc123", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("abc123", "Bracketed")
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        for raw in ("<abc123>", "[abc123]", '"abc123"', "'abc123'"):
            event = _make_event(text=f"/resume {raw}")
            runner = _make_runner(
                session_db=db,
                current_session_id="current_session_001",
                event=event,
            )
            result = await runner._handle_resume_command(event)
            # Either the session was resumed (and we get a "Resumed" / "Already on" reply)
            # or it was found-then-redirected. Failure mode = "No session found matching '<abc123>'".
            assert "abc123" not in str(result) or "not found" not in str(result).lower(), (
                f"bracket stripping failed for {raw!r}: gateway returned {result!r}"
            )
        db.close()

    @pytest.mark.asyncio
    async def test_resume_resolves_by_session_id(self, tmp_path):
        """The gateway should accept a bare session ID, not just a title.

        Before this fix, /resume in the gateway only called
        ``resolve_session_by_title``, so ``/resume <session_id>`` always
        returned "Session not found" even for valid IDs.
        """
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("unnamed_session_xyz", "telegram", user_id="12345", chat_id="67890")
        # Deliberately no title set — this session can ONLY be resolved by ID.
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        event = _make_event(text="/resume unnamed_session_xyz")
        runner = _make_runner(
            session_db=db,
            current_session_id="current_session_001",
            event=event,
        )
        result = await runner._handle_resume_command(event)

        # Should NOT be the not-found error.
        assert "not found" not in str(result).lower(), (
            f"session-id lookup failed: {result!r}"
        )
        db.close()



class TestHandleSessionsCommand:
    """Tests for GatewayRunner._handle_sessions_command."""

    @pytest.mark.asyncio
    async def test_sessions_command_lists_current_platform_sessions(self, tmp_path):
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("tg_session", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("tg_session", "Telegram Work")
        db.create_session("discord_session", "discord")
        db.set_session_title("discord_session", "Discord Work")

        event = _make_event(text="/sessions")
        runner = _make_runner(session_db=db, event=event)

        result = await runner._handle_sessions_command(event)

        assert "Sessions" in result
        assert "Telegram Work" in result
        assert "tg_session" in result
        assert "Discord Work" not in result
        db.close()

    @pytest.mark.asyncio
    async def test_sessions_all_does_not_leak_cross_origin_for_non_admin(self, tmp_path):
        """`/sessions all` from a non-admin caller must stay scoped to the
        caller's own origin — it must NOT enumerate other origins' sessions
        (the enumeration half of the /resume IDOR). Cross-origin listing is
        gated behind an explicitly-configured admin, which the default test
        config is not."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("tg_named", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("tg_named", "Telegram Work")
        db.create_session("discord_unnamed", "discord")  # other origin
        db.append_message("discord_unnamed", "user", "discord first prompt")

        event = _make_event(text="/sessions all full")
        runner = _make_runner(session_db=db, event=event)

        result = await runner._handle_sessions_command(event)

        # Caller's own (telegram) session is shown; the cross-origin (discord)
        # session is NOT leaked even with `all`.
        assert "Telegram Work" in result
        assert "discord_unnamed" not in result
        assert "Discord" not in result
        db.close()

    @pytest.mark.asyncio
    async def test_resume_blocks_cross_user_and_unowned_rows(self, tmp_path):
        """An identity-bearing caller cannot resume a session it can't prove it
        owns: a row owned by a different user, or a same-platform row with no
        recorded owner (NULL user_id) must both be denied (IDOR)."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("victim_other_uid", "telegram", user_id="99999")
        db.set_session_title("victim_other_uid", "Other User")
        db.create_session("victim_missing_uid", "telegram")  # NULL owner
        db.set_session_title("victim_missing_uid", "Unowned")
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        for name in ("Other User", "victim_other_uid", "Unowned", "victim_missing_uid"):
            event = _make_event(text=f"/resume {name}")
            runner = _make_runner(session_db=db, current_session_id="current_session_001",
                                  event=event)
            result = await runner._handle_resume_command(event)
            runner.session_store.switch_session.assert_not_called()
            assert "Resumed" not in result, name
        db.close()

    @pytest.mark.asyncio
    async def test_resume_blocks_blank_source_same_uid_row(self, tmp_path):
        """A persisted row whose `source` is blank/legacy cannot prove it shares
        the caller's platform, so user_id equality alone must NOT authorize a
        resume — the blank source fails closed exactly like a missing user_id
        (IDOR regression: an identified caller could otherwise bind to an
        unproven-origin transcript)."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("blank_source_same_uid", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("blank_source_same_uid", "Blank Source Same UID")
        # Simulate a malformed/legacy row that does not record its origin.
        db._conn.execute(
            "UPDATE sessions SET source = '' WHERE id = ?", ("blank_source_same_uid",)
        )
        db._conn.commit()
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        for name in ("Blank Source Same UID", "blank_source_same_uid"):
            event = _make_event(text=f"/resume {name}")
            runner = _make_runner(session_db=db, current_session_id="current_session_001",
                                  event=event)
            result = await runner._handle_resume_command(event)
            runner.session_store.switch_session.assert_not_called()
            assert "Resumed" not in result, name
        db.close()

    @pytest.mark.asyncio
    async def test_resume_blocks_no_identity_caller_on_persisted_row(self, tmp_path):
        """A caller with no user_id must not resume a persisted row on
        same-platform alone: the row has no chat_id to prove ownership, so a
        Telegram group caller in chat-a (user_id=None) cannot bind to a row
        owned by another chat/user (IDOR regression for the no-identity branch)."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("victim_chat_b_uid", "telegram", user_id="victim")
        db.set_session_title("victim_chat_b_uid", "Victim Chat B")
        db.create_session("current_session_001", "telegram")

        for name in ("Victim Chat B", "victim_chat_b_uid"):
            event = _make_event(text=f"/resume {name}", user_id=None,
                                chat_id="chat-a")
            event.source.chat_type = "group"
            runner = _make_runner(session_db=db, current_session_id="current_session_001",
                                  event=event)
            result = await runner._handle_resume_command(event)
            runner.session_store.switch_session.assert_not_called()
            assert "Resumed" not in result, name
        db.close()

    @pytest.mark.asyncio
    async def test_resume_target_allowed_blocks_no_identity_persisted(self, tmp_path):
        """Unit-level: the persisted-row fallback fails closed for an
        identity-less caller (no live origin resolvable)."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("victim_chat_b_uid", "telegram", user_id="victim")
        runner = _make_runner(session_db=db)
        runner._gateway_session_origin_for_id = lambda sid: None  # inactive/persisted-only
        caller = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-a",
                               chat_type="group", user_id=None)
        assert await runner._resume_target_allowed(caller, "victim_chat_b_uid",
                                             allow_override=False) is False
        db.close()

    @pytest.mark.asyncio
    async def test_resume_blocks_same_user_different_chat(self, tmp_path):
        """egilewski/CodeRabbit probe: the SAME user must not move a persisted
        transcript from another chat into the current one. The row records its
        records origin chat_id, so a chat-a caller cannot resume a chat-b row even with
        a matching user_id (persisted-row chat-scope proof)."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("same_user_chat_b", "telegram", user_id="12345",
                          chat_id="chat-b")
        db.set_session_title("same_user_chat_b", "Same User Chat B")
        db.create_session("current_session_001", "telegram", user_id="12345",
                          chat_id="chat-a")

        for name in ("Same User Chat B", "same_user_chat_b"):
            event = _make_event(text=f"/resume {name}", user_id="12345",
                                chat_id="chat-a")
            event.source.chat_type = "group"
            runner = _make_runner(session_db=db, current_session_id="current_session_001",
                                  event=event)
            result = await runner._handle_resume_command(event)
            runner.session_store.switch_session.assert_not_called()
            assert "Resumed" not in result, name
        db.close()

    @pytest.mark.asyncio
    async def test_resume_target_allowed_chat_scope(self, tmp_path):
        """Unit-level: identity-bearing persisted fallback requires the row's
        origin chat (and thread) to match the caller's."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("row_chat_a", "telegram", user_id="12345",
                          chat_id="chat-a")
        db.create_session("row_chat_b", "telegram", user_id="12345",
                          chat_id="chat-b")
        db.create_session("row_legacy_nochat", "telegram", user_id="12345")  # NULL chat
        runner = _make_runner(session_db=db)
        runner._gateway_session_origin_for_id = lambda sid: None  # persisted-only
        caller = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-a",
                               chat_type="group", user_id="12345")
        # Same chat → allowed; different chat → blocked; legacy NULL-chat → blocked.
        assert await runner._resume_target_allowed(caller, "row_chat_a", allow_override=False) is True
        assert await runner._resume_target_allowed(caller, "row_chat_b", allow_override=False) is False
        assert await runner._resume_target_allowed(caller, "row_legacy_nochat", allow_override=False) is False
        # egilewski/CodeRabbit probe: a GROUP caller that itself has no chat_id
        # must NOT resume a legacy NULL-chat row just because both normalize to
        # "" — a non-DM session is keyed by chat_id, so blank == no provenance.
        blank_caller = SessionSource(platform=Platform.TELEGRAM, chat_id=None,
                                     chat_type="group", user_id="12345")
        assert await runner._resume_target_allowed(blank_caller, "row_legacy_nochat",
                                             allow_override=False) is False
        db.close()

    @pytest.mark.asyncio
    async def test_resume_target_allowed_dm_no_chat_id_scopes_by_user(self, tmp_path):
        """A DM is keyed on user_id; a no-chat_id DM row is resumable by the same
        user (chat_id legitimately absent on both sides), unlike a group row."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("dm_row", "telegram", user_id="12345")  # DM, no chat_id
        runner = _make_runner(session_db=db)
        runner._gateway_session_origin_for_id = lambda sid: None  # persisted-only
        same = SessionSource(platform=Platform.TELEGRAM, chat_id=None,
                             chat_type="dm", user_id="12345")
        other = SessionSource(platform=Platform.TELEGRAM, chat_id=None,
                              chat_type="dm", user_id="99999")
        assert await runner._resume_target_allowed(same, "dm_row", allow_override=False) is True
        assert await runner._resume_target_allowed(other, "dm_row", allow_override=False) is False
        db.close()

    @pytest.mark.asyncio
    async def test_resume_target_allowed_shared_group_no_user_match(self, tmp_path):
        """egilewski probe: with group_sessions_per_user=False a non-DM group
        session is shared, so a co-member (different user_id) in the SAME chat
        may resume it — same-chat/thread proof is sufficient, user equality is
        not required. Per-user groups (default) still require the same owner."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("shared_group_row", "telegram", user_id="bob",
                          chat_id="shared-chat", chat_type="group")
        runner = _make_runner(session_db=db)
        runner._gateway_session_origin_for_id = lambda sid: None  # persisted-only
        alice = SessionSource(platform=Platform.TELEGRAM, chat_id="shared-chat",
                              chat_type="group", user_id="alice")

        # Shared group → Alice may resume Bob's row in the same chat.
        runner.config.group_sessions_per_user = False
        assert await runner._resume_target_allowed(alice, "shared_group_row",
                                                   allow_override=False) is True
        # Per-user group → Alice must NOT resume Bob's row (IDOR preserved).
        runner.config.group_sessions_per_user = True
        assert await runner._resume_target_allowed(alice, "shared_group_row",
                                                   allow_override=False) is False
        # A different chat is still blocked even when shared.
        runner.config.group_sessions_per_user = False
        other_chat = SessionSource(platform=Platform.TELEGRAM, chat_id="other-chat",
                                   chat_type="group", user_id="alice")
        assert await runner._resume_target_allowed(other_chat, "shared_group_row",
                                                   allow_override=False) is False
        db.close()

    @pytest.mark.asyncio
    async def test_resume_persisted_fallback_fails_closed_on_user_id_alt(self, tmp_path):
        """egilewski/CodeRabbit probe: Signal/Feishu key the session participant
        on ``user_id_alt or user_id`` (build_session_key), but the sessions table
        stores only user_id. So a persisted per-user row that a caller shares the
        user_id of — but NOT the user_id_alt — maps to a DIFFERENT live session
        key; the persisted fallback must NOT match it on user_id alone (IDOR).

        The live-origin guard already compares user_id_alt correctly; here the
        target is persisted-only, so the fallback fails closed whenever the
        caller keys on user_id_alt and the row can't prove that participant."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        # Persisted rows carry only user_id (no user_id_alt column).
        db.create_session("victim_alt_group", "signal", user_id="+15550001111",
                          chat_id="signal-group", chat_type="group")
        db.create_session("victim_alt_dm", "signal", user_id="+15550001111")  # no chat_id
        runner = _make_runner(session_db=db)
        runner._gateway_session_origin_for_id = lambda sid: None  # persisted-only

        # Per-user group: attacker shares user_id but has a different user_id_alt
        # → different session key → must fail closed (was: allowed via user_id).
        attacker = SessionSource(platform=Platform.SIGNAL, chat_id="signal-group",
                                 chat_type="group", user_id="+15550001111",
                                 user_id_alt="attacker-uuid")
        assert await runner._resume_target_allowed(attacker, "victim_alt_group",
                                                   allow_override=False) is False
        # No-chat_id DM keyed purely on the participant: same block.
        dm_attacker = SessionSource(platform=Platform.SIGNAL, chat_id=None,
                                    chat_type="dm", user_id="+15550001111",
                                    user_id_alt="attacker-uuid")
        assert await runner._resume_target_allowed(dm_attacker, "victim_alt_dm",
                                                   allow_override=False) is False

        # Regression: a caller WITHOUT user_id_alt (Telegram-style, keyed on
        # user_id) still resumes its own persisted per-user group row.
        tg_db = SessionDB(db_path=tmp_path / "state_tg.db")
        tg_db.create_session("own_group", "telegram", user_id="12345",
                             chat_id="chat-a", chat_type="group")
        tg_runner = _make_runner(session_db=tg_db)
        tg_runner._gateway_session_origin_for_id = lambda sid: None
        tg_caller = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-a",
                                  chat_type="group", user_id="12345")
        assert await tg_runner._resume_target_allowed(tg_caller, "own_group",
                                                      allow_override=False) is True

        # Regression: an EXPLICITLY-shared group is unaffected — participant
        # scoping doesn't apply, so an alt-keyed co-member still resumes.
        runner.config.group_sessions_per_user = False
        assert await runner._resume_target_allowed(attacker, "victim_alt_group",
                                                   allow_override=False) is True
        db.close()
        tg_db.close()

    @pytest.mark.asyncio
    async def test_gateway_dispatches_sessions_command(self, tmp_path):
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("tg_session", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("tg_session", "Telegram Work")

        event = _make_event(text="/sessions")
        runner = _make_runner(session_db=db, event=event)
        runner._handle_sessions_command = AsyncMock(return_value="sessions output")

        result = await runner._handle_message(event)

        assert result == "sessions output"
        runner._handle_sessions_command.assert_awaited_once_with(event)
        db.close()


class TestSameOriginChatGroupScoping:
    """Live group sessions are per-user by default (group_sessions_per_user=True),
    so a co-member must not be able to resume another member's live group session
    via the live-origin branch of _resume_target_allowed (IDOR)."""

    @staticmethod
    def _src(user_id, *, chat_type="group", chat_id="guild-123",
             platform=Platform.DISCORD, user_id_alt=None, thread_id=None):
        return SessionSource(platform=platform, chat_id=chat_id,
                             chat_type=chat_type, user_id=user_id,
                             user_id_alt=user_id_alt, thread_id=thread_id)

    def test_blocks_cross_user_live_group_by_default(self):
        runner = _make_runner()
        assert runner._same_origin_chat(self._src("alice"), self._src("bob")) is False

    def test_allows_same_user_live_group(self):
        runner = _make_runner()
        assert runner._same_origin_chat(self._src("alice"), self._src("alice")) is True

    def test_allows_cross_user_when_group_explicitly_shared(self):
        runner = _make_runner()
        runner.config.group_sessions_per_user = False
        assert runner._same_origin_chat(self._src("alice"), self._src("bob")) is True

    def test_dm_cross_user_blocked_without_chat_id(self):
        # No-chat_id DM: build_session_key falls back to the participant id
        # (user_id_alt or user_id), so two different participants are different
        # origins and must not match. (With a chat_id present the DM key IS the
        # chat_id — see test_dm_same_chat_id_is_same_origin.)
        runner = _make_runner()
        a = self._src("alice", chat_type="dm", chat_id=None)
        b = self._src("bob", chat_type="dm", chat_id=None)
        assert runner._same_origin_chat(a, b) is False

    def test_dm_no_identity_no_chat_id_fails_closed(self):
        # teknium1 review: an identity-less no-chat_id DM must fail closed rather
        # than be treated as a shared origin.
        runner = _make_runner()
        a = self._src(None, chat_type="dm", chat_id=None)
        b = self._src(None, chat_type="dm", chat_id=None)
        assert runner._same_origin_chat(a, b) is False

    def test_dm_user_id_alt_mismatch_without_chat_id_blocked(self):
        # No-chat_id DM keyed on user_id_alt (Signal/Feishu): different alt ids
        # are different sessions even if user_id is absent/equal.
        runner = _make_runner()
        a = self._src(None, chat_type="dm", chat_id=None, user_id_alt="alice-alt")
        b = self._src(None, chat_type="dm", chat_id=None, user_id_alt="bob-alt")
        assert runner._same_origin_chat(a, b) is False

    def test_dm_same_chat_id_is_same_origin(self):
        # With a chat_id present, the DM session key is chat_id-only (no
        # participant), so an equal chat_id is a same-origin match — mirrors
        # build_session_key.
        runner = _make_runner()
        a = self._src("alice", chat_type="dm", chat_id="dm-1")
        b = self._src("alice", chat_type="dm", chat_id="dm-1")
        assert runner._same_origin_chat(a, b) is True

    @pytest.mark.asyncio
    async def test_resume_target_allowed_blocks_cross_user_live_group(self):
        """End-to-end via the live-origin branch: Alice cannot resume Bob's
        active group session in the same chat."""
        runner = _make_runner()
        bob = self._src("bob")
        runner._gateway_session_origin_for_id = lambda sid: bob
        assert await runner._resume_target_allowed(
            self._src("alice"), "bobs_live_sid", allow_override=False
        ) is False

    # --- thread scoping: thread_id is part of the session key, so a session in
    # one thread must never match a caller in another thread of the same chat,
    # even when threads are shared among participants by default. ---

    def test_blocks_cross_thread_same_user_same_chat(self):
        """Same user, same parent chat, different thread → different session."""
        runner = _make_runner()
        a = self._src("alice", thread_id="thread-A")
        b = self._src("alice", thread_id="thread-B")
        assert runner._same_origin_chat(a, b) is False

    def test_allows_same_thread_shared_participants(self):
        """Threads are shared by default (thread_sessions_per_user=False), so
        co-members in the SAME thread share the session."""
        runner = _make_runner()
        a = self._src("alice", thread_id="thread-A")
        b = self._src("bob", thread_id="thread-A")
        assert runner._same_origin_chat(a, b) is True

    def test_blocks_cross_thread_even_when_shared(self):
        """Cross-thread is blocked regardless of thread-sharing: sharing only
        applies WITHIN a thread, never across threads."""
        runner = _make_runner()
        a = self._src("alice", thread_id="thread-A")
        b = self._src("bob", thread_id="thread-B")
        assert runner._same_origin_chat(a, b) is False

    def test_blocks_thread_vs_no_thread(self):
        """A threaded origin must not match a non-threaded caller in the same
        parent chat (and vice versa)."""
        runner = _make_runner()
        threaded = self._src("alice", thread_id="thread-A")
        parent = self._src("alice", thread_id=None)
        assert runner._same_origin_chat(parent, threaded) is False
        assert runner._same_origin_chat(threaded, parent) is False


class TestResumeRowVisibleMatrixAllScoping:
    """Non-admin Matrix `/resume --all` must NOT enumerate every Matrix titled
    session: the cross-room listing short-circuit is admin-only, mirroring the
    non-Matrix branch. A non-admin `--all` falls back to same-room scoping."""

    @staticmethod
    def _matrix_src(chat_id="!room-a:hs", user_id="@alice:hs"):
        return SessionSource(platform=Platform.MATRIX, chat_id=chat_id,
                             chat_type="group", user_id=user_id)

    @pytest.mark.asyncio
    async def test_non_admin_all_does_not_expose_other_room(self):
        runner = _make_runner()
        runner._resume_caller_is_admin = lambda src: False
        # Titled row whose live origin is a DIFFERENT Matrix room.
        other_room = SessionSource(platform=Platform.MATRIX, chat_id="!room-b:hs",
                                   chat_type="group", user_id="@bob:hs")
        runner._gateway_session_origin_for_id = lambda sid: other_room
        row = {"id": "sid_other_room"}
        assert await runner._resume_row_visible(self._matrix_src(), row, allow_all=True) is False

    @pytest.mark.asyncio
    async def test_non_admin_all_still_shows_same_room(self):
        runner = _make_runner()
        runner._resume_caller_is_admin = lambda src: False
        same_room = SessionSource(platform=Platform.MATRIX, chat_id="!room-a:hs",
                                  chat_type="group", user_id="@bob:hs")
        runner._gateway_session_origin_for_id = lambda sid: same_room
        row = {"id": "sid_same_room"}
        assert await runner._resume_row_visible(self._matrix_src(), row, allow_all=True) is True

    @pytest.mark.asyncio
    async def test_admin_all_exposes_cross_room(self):
        runner = _make_runner()
        runner._resume_caller_is_admin = lambda src: True
        other_room = SessionSource(platform=Platform.MATRIX, chat_id="!room-b:hs",
                                   chat_type="group", user_id="@bob:hs")
        runner._gateway_session_origin_for_id = lambda sid: other_room
        row = {"id": "sid_other_room"}
        assert await runner._resume_row_visible(self._matrix_src(), row, allow_all=True) is True

    @pytest.mark.asyncio
    async def test_non_admin_all_fails_closed_on_unknown_origin(self):
        runner = _make_runner()
        runner._resume_caller_is_admin = lambda src: False
        runner._gateway_session_origin_for_id = lambda sid: None
        row = {"id": "sid_unknown"}
        assert await runner._resume_row_visible(self._matrix_src(), row, allow_all=True) is False


class TestSameMatrixRoomThreadScoping:
    """Matrix `/resume` (direct and listing) scopes by room AND thread: a live
    session in another thread of the same room is a different session
    (build_session_key appends thread_id), so a caller in thread A must not
    resume/enumerate a target whose origin is in thread B. Non-threaded rooms
    keep room-level sharing unchanged."""

    @staticmethod
    def _msrc(chat_id="!room-a:hs", user_id="@alice:hs", thread_id=None):
        return SessionSource(platform=Platform.MATRIX, chat_id=chat_id,
                             chat_type="group", user_id=user_id, thread_id=thread_id)

    def test_same_room_no_thread_still_shared(self):
        runner = _make_runner()
        a = self._msrc(user_id="@alice:hs")
        b = self._msrc(user_id="@bob:hs")
        assert runner._same_matrix_room(a, b) is True

    def test_same_room_same_thread_shared(self):
        runner = _make_runner()
        a = self._msrc(user_id="@alice:hs", thread_id="thr-1")
        b = self._msrc(user_id="@bob:hs", thread_id="thr-1")
        assert runner._same_matrix_room(a, b) is True

    def test_cross_thread_same_room_blocked(self):
        """The reviewer's probe: caller in thread-a, target origin in thread-b
        of the same room → must not match."""
        runner = _make_runner()
        caller = self._msrc(thread_id="thread-a")
        victim_origin = self._msrc(thread_id="thread-b")
        assert runner._same_matrix_room(caller, victim_origin) is False

    def test_thread_vs_no_thread_blocked(self):
        runner = _make_runner()
        threaded = self._msrc(thread_id="thread-a")
        room_level = self._msrc(thread_id=None)
        assert runner._same_matrix_room(threaded, room_level) is False
        assert runner._same_matrix_room(room_level, threaded) is False

    @pytest.mark.asyncio
    async def test_resume_row_visible_blocks_cross_thread(self):
        """End-to-end through the Matrix listing guard."""
        runner = _make_runner()
        runner._resume_caller_is_admin = lambda src: False
        origin_thread_b = self._msrc(thread_id="thread-b")
        runner._gateway_session_origin_for_id = lambda sid: origin_thread_b
        row = {"id": "sid_thread_b"}
        caller_thread_a = self._msrc(thread_id="thread-a")
        assert await runner._resume_row_visible(caller_thread_a, row, allow_all=False) is False

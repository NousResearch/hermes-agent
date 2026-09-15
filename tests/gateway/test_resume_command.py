"""Tests for /resume gateway slash command.

Tests the _handle_resume_command handler (switch to a previously-named session)
across gateway messenger platforms.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
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


def _make_matrix_event(
    text="/resume",
    *,
    user_id="@alice:example.org",
    chat_id="!room-a:example.org",
    thread_id=None,
):
    event = _make_event(
        text=text,
        platform=Platform.MATRIX,
        user_id=user_id,
        chat_id=chat_id,
    )
    event.source.chat_type = "group"
    event.source.chat_name = "Current Matrix Room"
    event.source.thread_id = thread_id
    return event


def _record_gateway_origin(
    db,
    session_id,
    source,
    *,
    group_sessions_per_user=True,
    thread_sessions_per_user=False,
):
    """Persist the trusted full gateway provenance written by SessionStore."""
    db.record_gateway_session_peer(
        session_id,
        source=source.platform.value,
        user_id=source.user_id,
        session_key=build_session_key(
            source,
            group_sessions_per_user=group_sessions_per_user,
            thread_sessions_per_user=thread_sessions_per_user,
            profile=source.profile,
        ),
        chat_id=source.chat_id,
        chat_type=source.chat_type,
        thread_id=source.thread_id,
        display_name=source.chat_name,
        origin_json=json.dumps(source.to_dict()),
    )


def _session_key_for_event(event):
    """Get the session key that build_session_key produces for an event."""
    return build_session_key(event.source)


def _make_runner(
    session_db=None,
    current_session_id="current_session_001",
    event=None,
    *,
    persist_event_origin=True,
):
    """Create a bare GatewayRunner with a mock session store and optional DB."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = SimpleNamespace(platforms={})
    runner._voice_mode = {}
    # Gateway holds the async facade; the slash handlers await it. Tests may
    # explicitly mirror SessionStore persistence for completely blank legacy
    # fixture rows, but never overwrite existing or malformed provenance.
    if session_db is not None:
        if event is not None and persist_event_origin:
            rows = session_db._conn.execute(
                "SELECT id, source, user_id, chat_id, chat_type, thread_id, "
                "session_key, origin_json FROM sessions"
            ).fetchall()
            for row in rows:
                if row["origin_json"] is not None:
                    continue
                if (
                    str(row["source"] or "") == event.source.platform.value
                    and str(row["user_id"] or "") == str(event.source.user_id or "")
                    and str(row["chat_id"] or "") == str(event.source.chat_id or "")
                ):
                    persisted_source = SessionSource.from_dict(event.source.to_dict())
                    if row["chat_type"]:
                        persisted_source.chat_type = row["chat_type"]
                    if row["thread_id"]:
                        persisted_source.thread_id = row["thread_id"]
                    _record_gateway_origin(session_db, row["id"], persisted_source)
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
        event = _make_event(text="/resume")
        lane_key = _session_key_for_event(event)
        db.create_session(
            "sess_001", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.create_session(
            "sess_002", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("sess_001", "Research")
        db.set_session_title("sess_002", "Coding")

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
    async def test_resume_all_nonadmin_downgrade_is_announced(self, tmp_path):
        """A non-admin `/resume --all` must say the widening was declined."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/resume --all")
        lane_key = _session_key_for_event(event)
        db.create_session(
            "sess_001", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("sess_001", "Research")

        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_resume_command(event)
        assert "Research" in result
        assert "requires a configured admin" in result
        db.close()

    @pytest.mark.asyncio
    async def test_resume_plain_listing_has_no_scope_notice(self, tmp_path):
        """No downgrade notice when `--all` wasn't requested."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/resume")
        lane_key = _session_key_for_event(event)
        db.create_session(
            "sess_001", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("sess_001", "Research")

        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_resume_command(event)
        assert "Research" in result
        assert "requires a configured admin" not in result
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
    async def test_resume_clears_last_resolved_model(self, tmp_path):
        """Resume must also clear the resumed chat's cached last-resolved
        model, so the restored conversation re-resolves from current config
        instead of a value cached before the switch (mirrors /new and the
        compression-exhausted auto-reset, #58403), while leaving other
        chats' cache entries intact."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("old_session_abc", "telegram", user_id="12345", chat_id="67890")
        db.set_session_title("old_session_abc", "My Project")
        db.create_session("current_session_001", "telegram", user_id="12345", chat_id="67890")

        event = _make_event(text="/resume My Project")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        key = _session_key_for_event(event)
        runner._last_resolved_model = {
            key: "gpt-5",
            "agent:main:telegram:dm:other": "keep-me",
        }

        result = await runner._handle_resume_command(event)

        assert "Resumed" in result
        assert key not in runner._last_resolved_model
        assert runner._last_resolved_model["agent:main:telegram:dm:other"] == "keep-me"
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
    async def test_bare_resume_lists_exact_lane_before_limit(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/resume")
        lane_key = _session_key_for_event(event)
        for i in range(3):
            sid = f"lane_{i}"
            db.create_session(
                sid, "telegram", session_key=lane_key,
                user_id="12345", chat_id="67890",
            )
            db.set_session_title(sid, f"Lane Work {i}")
        for i in range(12):
            sid = f"foreign_{i}"
            db.create_session(
                sid, "telegram",
                session_key=f"agent:main:telegram:dm:foreign-{i}",
                user_id=f"foreign-user-{i}", chat_id=f"foreign-{i}",
            )
            db.set_session_title(sid, f"Foreign Work {i}")

        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_resume_command(event)

        assert "Lane Work 0" in result
        assert "Lane Work 1" in result
        assert "Lane Work 2" in result
        assert "Foreign Work" not in result
        db.close()

    @pytest.mark.asyncio
    async def test_bare_resume_admin_all_preserves_same_platform_widening(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/resume --all")
        db.create_session(
            "other_lane", "telegram",
            session_key="agent:main:telegram:dm:other",
            user_id="other-user", chat_id="other",
        )
        db.set_session_title("other_lane", "Other Lane Work")
        foreign = _make_event(platform=Platform.DISCORD, user_id="victim", chat_id="other").source
        TestCanonicalResumeAuthorization._create(db, "foreign_platform", foreign, "Foreign Platform Work")
        db.append_message("foreign_platform", "user", "FOREIGN_PLATFORM_PREVIEW")

        runner = _make_runner(session_db=db, event=event)
        runner.config.platforms[Platform.TELEGRAM] = PlatformConfig(extra={
            "allow_admin_from": [event.source.user_id],
        })
        assert runner._resume_caller_is_admin(event.source)
        result = await runner._handle_resume_command(event)

        assert "Other Lane Work" in result
        for value in ("foreign_platform", "Foreign Platform Work", "FOREIGN_PLATFORM_PREVIEW"):
            assert value not in result
        db.close()

    @pytest.mark.asyncio
    async def test_numeric_resume_fallback_uses_exact_lane_candidates(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/resume 2")
        lane_key = _session_key_for_event(event)
        db.create_session(
            "lane_older", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("lane_older", "Lane Older")
        db.create_session(
            "lane_newer", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("lane_newer", "Lane Newer")
        for i in range(12):
            sid = f"foreign_{i}"
            db.create_session(
                sid, "telegram",
                session_key=f"agent:main:telegram:dm:foreign-{i}",
                user_id=f"foreign-user-{i}", chat_id=f"foreign-{i}",
            )
            db.set_session_title(sid, f"Foreign Work {i}")
        db.create_session(
            "current_session_001", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )

        runner = _make_runner(
            session_db=db, current_session_id="current_session_001", event=event
        )
        result = await runner._handle_resume_command(event)

        assert "Resumed" in result
        runner.session_store.switch_session.assert_called_once()
        assert runner.session_store.switch_session.call_args[0][1] == "lane_older"
        db.close()

    @pytest.mark.asyncio
    async def test_bare_resume_normalizes_telegram_lobby_source_to_bound_topic(
        self, tmp_path
    ):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/resume")
        topic_source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            chat_type="dm",
            thread_id="topic-42",
        )
        topic_key = build_session_key(topic_source)
        db.create_session(
            "topic_session", "telegram", session_key=topic_key,
            user_id="12345", chat_id="67890", chat_type="dm",
            thread_id="topic-42",
        )
        db.set_session_title("topic_session", "Recovered Topic Work")
        db.enable_telegram_topic_mode(chat_id="67890", user_id="12345")
        db.bind_telegram_topic(
            chat_id="67890",
            thread_id="topic-42",
            user_id="12345",
            session_key=topic_key,
            session_id="topic_session",
        )
        lobby_key = _session_key_for_event(event)
        db.create_session(
            "lobby_session", "telegram", session_key=lobby_key,
            user_id="12345", chat_id="67890", chat_type="dm",
        )
        db.set_session_title("lobby_session", "Lobby Work")

        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_resume_command(event)

        assert "Recovered Topic Work" in result
        assert "Lobby Work" not in result
        db.close()




class TestCompleteAuthorizedListing:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("flags", [
        "--all", "--full", "--all --full", "--full --all",
    ])
    async def test_resume_flags_keep_native_semantics_without_sessions_delegation(self, tmp_path, flags):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(f"/resume {flags}")
        TestCanonicalResumeAuthorization._create(db, "full_target", event.source, "--full")
        runner = _make_runner(session_db=db, event=event)
        runner._handle_sessions_command = AsyncMock(return_value="wrong listing path")
        try:
            result = await runner._handle_resume_command(event)
            if flags == "--all":
                assert "Named Sessions" in result and "--full" in result
                assert "1." in result and "/resume 1" in result
                runner.session_store.switch_session.assert_not_called()
            else:
                assert "Resumed" in result
                assert runner.session_store.switch_session.call_args.args[1] == "full_target"
            runner._handle_sessions_command.assert_not_awaited()
            assert event.text == f"/resume {flags}"
        finally:
            db.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("listing_args", ["all", "full", "all full"])
    @pytest.mark.parametrize("admin", [False, True])
    async def test_sessions_keeps_authorization_and_formatting(
        self, tmp_path, listing_args, admin
    ):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(f"/sessions {listing_args}")
        create = TestCanonicalResumeAuthorization._create
        create(db, "own_named", event.source, "Own Work")
        create(db, "own_unnamed", event.source)
        db.append_message("own_unnamed", "user", "own preview")
        foreign = _make_event(platform=Platform.DISCORD, user_id="victim", chat_id="other").source
        create(db, "foreign_named", foreign, "Foreign Secret Title")
        db.append_message("foreign_named", "user", "foreign secret preview")
        runner = _make_runner(session_db=db, current_session_id="own_named", event=event)
        runner.config.platforms[Platform.TELEGRAM] = PlatformConfig(extra={
            "allow_admin_from": [event.source.user_id if admin else "operator"],
        })
        assert runner._resume_caller_is_admin(event.source) is admin
        try:
            actual = await runner._handle_sessions_command(event)
            assert "**Own Work** (current)" in actual
            if "full" in listing_args.split():
                assert "own_unnamed" in actual and "own preview" in actual
            else:
                assert "own_unnamed" not in actual
            for secret in ("foreign_named", "Foreign Secret Title", "foreign secret preview"):
                assert (secret in actual) is (admin and "all" in listing_args.split())
            assert ("requires a configured admin" in actual) is (not admin and "all" in listing_args.split())
            runner.session_store.switch_session.assert_not_called()
        finally:
            db.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("target", ["all", '"all"', '"Project all"', "title_all", "--all title_all"])
    async def test_resume_all_without_dashes_remains_a_title(self, tmp_path, target):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(f"/resume {target}")
        title = "Project all" if target == '"Project all"' else "all"
        TestCanonicalResumeAuthorization._create(db, "title_all", event.source, title)
        runner = _make_runner(session_db=db, event=event)
        runner._handle_sessions_command = AsyncMock(return_value="wrong listing path")
        try:
            result = await runner._handle_resume_command(event)
            assert "Resumed" in result
            assert runner.session_store.switch_session.call_args.args[1] == "title_all"
            runner._handle_sessions_command.assert_not_awaited()
        finally:
            db.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("target", ["--unknown", "--full own-target", "--all --unknown"])
    async def test_mixed_listing_flags_keep_direct_target_semantics(self, target):
        event = _make_event(f"/resume {target}")
        runner = _make_runner(event=event)
        runner._session_db = AsyncMock()
        runner._telegram_topic_mode_enabled = lambda source: False
        runner._session_db.get_session.return_value = None
        runner._session_db.list_session_title_candidates.return_value = []
        runner._handle_sessions_command = AsyncMock(return_value="wrong listing path")
        result = await runner._handle_resume_command(event)
        assert "No session found" in result
        runner._session_db.get_session.assert_awaited_once_with(target.replace("--all ", ""))
        runner._handle_sessions_command.assert_not_awaited()
        runner.session_store.switch_session.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("command", [
        "/resume", "/sessions", "/sessions full", "/sessions search needle", "/sessions --full",
    ])
    async def test_resume_and_sessions_paginate_past_newer_foreign_rows(self, tmp_path, command):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(command)
        create = TestCanonicalResumeAuthorization._create
        create(db, "own_older", event.source, "Needle Older")
        create(db, "own_newer", event.source, "Needle Newer")
        foreign = _make_event(user_id="victim", chat_id="other").source
        for i in range(65):
            # Roots pass the native lane SQL; projected foreign tips must fail
            # canonical root/tip authorization AFTER the database limit.
            root, tip = f"root_{i}", f"foreign_tip_{i}"
            create(db, root, event.source, f"Needle Private {i}")
            db.end_session(root, "compression")
            create(db, tip, foreign, parent=root)
            db.append_message(tip, "user", "PRIVATE_PREVIEW")
        runner = _make_runner(session_db=db, event=event, persist_event_origin=False)
        runner._gateway_session_origin_for_id = lambda sid: None
        lane = _session_key_for_event(event)
        try:
            first = db.list_sessions_rich(source="telegram", session_key=lane, limit=50)
            assert len(first) == 50
            assert all(row.get("_lineage_root_id", "").startswith("root_") for row in first)
            assert all(row["session_key"] == lane for row in first)
            assert not await runner._resume_row_visible(event.source, first[0], False)
            result = await (runner._handle_resume_command(event) if command.startswith("/resume")
                            else runner._handle_sessions_command(event))
            assert result.index("Needle Newer") < result.index("Needle Older")
            for secret in ("Needle Private", "foreign_tip_", "PRIVATE_PREVIEW"):
                assert secret not in result
            # Numeric choices use exactly the bare /resume display order.
            numbered = await runner._handle_resume_command(_make_event("/resume 2"))
            assert "Resumed" in numbered
            assert runner.session_store.switch_session.call_args.args[1] == "own_older"
        finally:
            db.close()


class TestHandleSessionsCommand:
    """Tests for GatewayRunner._handle_sessions_command."""

    @pytest.mark.asyncio
    async def test_sessions_full_keeps_legacy_reset_child_after_parent_resume(
        self, tmp_path
    ):
        import json

        from gateway.config import GatewayConfig
        from gateway.session import AsyncSessionStore, SessionStore
        from hermes_state import AsyncSessionDB

        event = _make_event(text="/sessions full")
        store = SessionStore(
            sessions_dir=tmp_path / "sessions",
            config=GatewayConfig(),
        )
        db = store._db
        assert db is not None

        root = store.get_or_create_session(event.source)
        root_id = root.session_id
        db.set_session_title(root_id, "Legacy reset parent")
        child = store.reset_session(root.session_key)
        assert child is not None
        child_id = child.session_id
        db.set_session_title(child_id, "Legacy reset child")
        # Reproduce the on-disk shape from before _reset_from existed.
        db._conn.execute(
            "UPDATE sessions SET model_config = NULL WHERE id = ?",
            (child_id,),
        )
        db._conn.commit()

        runner = _make_runner(session_db=None, event=event)
        runner.session_store = store
        runner._async_session_store = AsyncSessionStore(store)
        runner._session_db = AsyncSessionDB(db)

        before_resume = await runner._handle_sessions_command(event)
        assert "Legacy reset parent" in before_resume

        switched = store.switch_session(root.session_key, root_id)
        assert switched is not None
        after_resume = await runner._handle_sessions_command(event)

        assert "Legacy reset child" in after_resume
        # The parent is now the CURRENT session: since #68547 it stays in the
        # listing with a "(current)" marker instead of being hidden.
        assert "**Legacy reset parent** (current)" in after_resume
        child_row = db.get_session(child_id)
        assert child_row is not None
        assert json.loads(child_row["model_config"])["_reset_from"] == root_id
        db.close()

    @pytest.mark.asyncio
    async def test_sessions_full_lists_conversations_created_by_gateway_resets(
        self, tmp_path
    ):
        import json

        from gateway.config import GatewayConfig
        from gateway.session import AsyncSessionStore, SessionStore
        from hermes_state import AsyncSessionDB

        event = _make_event(text="/sessions full")
        store = SessionStore(
            sessions_dir=tmp_path / "sessions",
            config=GatewayConfig(),
        )
        db = store._db
        assert db is not None

        entry = store.get_or_create_session(event.source)
        db.set_session_title(entry.session_id, "Greeting via Telegram")
        for title in (
            "Store memories with priority",
            "Extract AI news to Telegram",
            "Current Telegram work",
        ):
            previous_id = entry.session_id
            entry = store.reset_session(entry.session_key)
            assert entry is not None
            db.set_session_title(entry.session_id, title)
            reset_row = db.get_session(entry.session_id)
            assert reset_row is not None
            assert json.loads(reset_row["model_config"])["_reset_from"] == previous_id

        # The gateway creates the identity row before the agent exists. Its
        # first-turn create_session upsert must enrich the marker-only config,
        # while later bare/retry upserts must not replace the established data.
        db.create_session(
            entry.session_id,
            "telegram",
            model_config={"max_iterations": 60},
        )
        enriched = json.loads(db.get_session(entry.session_id)["model_config"])
        assert enriched == {
            "max_iterations": 60,
            "_reset_from": previous_id,
        }
        db.create_session(
            entry.session_id,
            "telegram",
            model_config={"max_iterations": 999},
        )
        assert json.loads(db.get_session(entry.session_id)["model_config"]) == enriched

        runner = _make_runner(session_db=None, event=event)
        runner.session_store = store
        runner._async_session_store = AsyncSessionStore(store)
        runner._session_db = AsyncSessionDB(db)

        result = await runner._handle_sessions_command(event)

        assert "Greeting via Telegram" in result
        assert "Store memories with priority" in result
        assert "Extract AI news to Telegram" in result
        # The live tip is the current session — listed with the marker since
        # #68547 rather than hidden.
        assert "**Current Telegram work** (current)" in result
        db.close()

    @pytest.mark.asyncio
    async def test_sessions_busy_platform_lists_exact_lane_and_excludes_current_tip(
        self, tmp_path
    ):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions")
        lane_key = _session_key_for_event(event)
        for i in range(11):
            sid = f"lane_root_{i}"
            db.create_session(
                sid, "telegram", session_key=lane_key,
                user_id="12345", chat_id="67890",
            )
            db.set_session_title(sid, f"Lane Work {i}")

        db.create_session(
            "current_root", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("current_root", "Current compressed root")
        db.end_session("current_root", "compression")
        db.create_session(
            "current_tip", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890", parent_session_id="current_root",
        )
        db.set_session_title("current_tip", "Current compressed tip")

        for i in range(60):
            sid = f"foreign_{i}"
            db.create_session(
                sid, "telegram",
                session_key=f"agent:main:telegram:dm:foreign-{i}",
                user_id=f"foreign-user-{i}", chat_id=f"foreign-{i}",
            )
            db.set_session_title(sid, f"Foreign Work {i}")

        runner = _make_runner(
            session_db=db, current_session_id="current_tip", event=event
        )
        result = await runner._handle_sessions_command(event)

        # The current tip now occupies one of the 10 slots with a marker
        # (#68547) instead of being hidden; its compressed-away root stays out.
        assert "**Current compressed tip** (current)" in result
        assert result.count("Lane Work") == 9
        assert "`lane_root_2`" in result
        assert "`lane_root_1`" not in result
        assert "`lane_root_0`" not in result
        assert "Foreign Work" not in result
        assert "current_root" not in result
        db.close()

    @pytest.mark.asyncio
    async def test_sessions_all_nonadmin_downgrade_is_announced(self, tmp_path):
        """A non-admin `/sessions all` must say the widening was declined."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions all")
        lane_key = _session_key_for_event(event)
        db.create_session(
            "sess_local", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("sess_local", "Local Work")

        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_sessions_command(event)

        assert "Local Work" in result
        assert "requires a configured admin" in result
        db.close()

    @pytest.mark.asyncio
    async def test_sessions_plain_listing_has_no_scope_notice(self, tmp_path):
        """No notice when the caller never asked for `all`."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions")
        lane_key = _session_key_for_event(event)
        db.create_session(
            "sess_local", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("sess_local", "Local Work")

        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_sessions_command(event)

        assert "Local Work" in result
        assert "requires a configured admin" not in result
        db.close()

    @pytest.mark.asyncio
    async def test_sessions_admin_all_preserves_cross_origin_widening(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions all")
        lane_key = _session_key_for_event(event)
        db.create_session(
            "tg_named", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("tg_named", "Telegram Work")
        db.create_session(
            "discord_named", "discord",
            session_key="agent:main:discord:dm:other",
            user_id="other-user", chat_id="other",
        )
        db.set_session_title("discord_named", "Discord Work")

        runner = _make_runner(session_db=db, event=event)
        runner._resume_caller_is_admin = lambda _source: True
        result = await runner._handle_sessions_command(event)

        assert "Telegram Work" in result
        assert "Discord Work" in result
        db.close()

    @pytest.mark.asyncio
    async def test_sessions_normalizes_telegram_lobby_source_to_bound_topic(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions")
        topic_source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            chat_type="dm",
            thread_id="topic-42",
        )
        topic_key = build_session_key(topic_source)
        db.create_session(
            "topic_session", "telegram", session_key=topic_key,
            user_id="12345", chat_id="67890", chat_type="dm",
            thread_id="topic-42",
        )
        db.set_session_title("topic_session", "Recovered Topic Work")
        db.enable_telegram_topic_mode(chat_id="67890", user_id="12345")
        db.bind_telegram_topic(
            chat_id="67890",
            thread_id="topic-42",
            user_id="12345",
            session_key=topic_key,
            session_id="topic_session",
        )
        lobby_key = _session_key_for_event(event)
        db.create_session(
            "lobby_session", "telegram", session_key=lobby_key,
            user_id="12345", chat_id="67890", chat_type="dm",
        )
        db.set_session_title("lobby_session", "Lobby Work")

        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_sessions_command(event)

        assert "Recovered Topic Work" in result
        assert "Lobby Work" not in result
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
        event = _make_event(text="/sessions all full")
        lane_key = _session_key_for_event(event)
        db.create_session(
            "tg_named", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("tg_named", "Telegram Work")
        db.create_session("discord_unnamed", "discord")  # other origin
        db.append_message("discord_unnamed", "user", "discord first prompt")

        runner = _make_runner(session_db=db, event=event)

        result = await runner._handle_sessions_command(event)

        # Caller's own (telegram) session is shown; the cross-origin (discord)
        # session is NOT leaked even with `all`.
        assert "Telegram Work" in result
        assert "discord_unnamed" not in result
        assert "Discord" not in result
        db.close()

    @pytest.mark.asyncio
    async def test_sessions_search_finds_older_titled_session(self, tmp_path):
        """`/sessions search <query>` matches titles beyond the recent-10 list
        and orders by activity, keeping the caller's own scope."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions search an94")
        lane_key = _session_key_for_event(event)
        # Bury the target under newer sessions so a plain listing misses it.
        db.create_session(
            "target_an94", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("target_an94", "AN-94 Prestige Barrel Build #2")
        for i in range(12):
            sid = f"filler_{i}"
            db.create_session(
                sid, "telegram", session_key=lane_key,
                user_id="12345", chat_id="67890",
            )
            db.set_session_title(sid, f"Filler {i}")

        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_sessions_command(event)

        assert "AN-94 Prestige Barrel Build #2" in result
        assert "target_an94" in result
        assert "Filler" not in result
        db.close()


    @pytest.mark.asyncio
    async def test_sessions_search_does_not_leak_other_users_sessions(self, tmp_path):
        """Search results honor the same owner-scoping guard as listing —
        a matching title owned by a different user/chat must not surface."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions search an94")
        lane_key = _session_key_for_event(event)
        db.create_session(
            "mine", "telegram", session_key=lane_key,
            user_id="12345", chat_id="67890",
        )
        db.set_session_title("mine", "AN-94 mine")
        db.create_session(
            "theirs", "telegram",
            session_key="agent:main:telegram:dm:55555",
            user_id="99999", chat_id="55555",
        )
        db.set_session_title("theirs", "AN-94 someone else's secret")

        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_sessions_command(event)

        assert "AN-94 mine" in result
        assert "theirs" not in result
        assert "secret" not in result
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
        runner._gateway_session_origin_for_id = lambda session_id: None  # persisted-only

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
        tg_caller = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-a",
                                  chat_type="group", user_id="12345")
        _record_gateway_origin(tg_db, "own_group", tg_caller)
        tg_runner = _make_runner(session_db=tg_db)
        tg_runner._gateway_session_origin_for_id = lambda session_id: None
        assert await tg_runner._resume_target_allowed(tg_caller, "own_group",
                                                      allow_override=False) is True

        # Regression: an EXPLICITLY-shared group is unaffected — participant
        # scoping doesn't apply, so an alt-keyed co-member still resumes.
        signal_owner = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="signal-group",
            chat_type="group",
            user_id="+155****1111",
        )
        _record_gateway_origin(
            db,
            "victim_alt_group",
            signal_owner,
            group_sessions_per_user=False,
        )
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


    def test_dm_cross_user_blocked_without_chat_id(self):
        # No-chat_id DM: build_session_key falls back to the participant id
        # (user_id_alt or user_id), so two different participants are different
        # origins and must not match. (With a chat_id present the DM key IS the
        # chat_id — see test_dm_same_chat_id_is_same_origin.)
        runner = _make_runner()
        a = self._src("alice", chat_type="dm", chat_id=None)
        b = self._src("bob", chat_type="dm", chat_id=None)
        assert runner._same_origin_chat(a, b) is False


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


    def test_allows_same_thread_shared_participants(self):
        """Threads are shared by default (thread_sessions_per_user=False), so
        co-members in the SAME thread share the session."""
        runner = _make_runner()
        a = self._src("alice", thread_id="thread-A")
        b = self._src("bob", thread_id="thread-A")
        assert runner._same_origin_chat(a, b) is True


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

    def test_same_room_no_thread_is_user_scoped_by_default(self):
        runner = _make_runner()
        a = self._msrc(user_id="@alice:hs")
        b = self._msrc(user_id="@bob:hs")
        assert runner._same_matrix_room(a, b) is False

    def test_same_room_no_thread_can_be_explicitly_shared(self):
        runner = _make_runner()
        runner.config.group_sessions_per_user = False
        a = self._msrc(user_id="@alice:hs")
        b = self._msrc(user_id="@bob:hs")
        assert runner._same_matrix_room(a, b) is True


    def test_cross_thread_same_room_blocked(self):
        """The reviewer's probe: caller in thread-a, target origin in thread-b
        of the same room → must not match."""
        runner = _make_runner()
        caller = self._msrc(thread_id="thread-a")
        victim_origin = self._msrc(thread_id="thread-b")
        assert runner._same_matrix_room(caller, victim_origin) is False

class TestCanonicalResumeAuthorization:
    """Canonical proof is shared by direct IDs, titles, listings and continuations."""

    @staticmethod
    def _create(db, sid, source, title=None, parent=None, **policy):
        db.create_session(
            sid, source.platform.value, user_id=source.user_id,
            chat_id=source.chat_id, chat_type=source.chat_type,
            thread_id=source.thread_id, parent_session_id=parent,
        )
        _record_gateway_origin(db, sid, source, **policy)
        if title:
            db.set_session_title(sid, title)

    @staticmethod
    def _guard_side_effects(runner):
        runner._clear_conversation_scope = MagicMock()
        runner._evict_cached_agent = MagicMock()
        runner._release_running_agent_state = MagicMock()

    @staticmethod
    def _assert_denied(runner, result):
        assert "No session found" in result
        runner.session_store.switch_session.assert_not_called()
        runner.session_store.get_or_create_session.assert_not_called()
        runner.session_store.load_transcript.assert_not_called()
        runner._clear_conversation_scope.assert_not_called()
        runner._evict_cached_agent.assert_not_called()
        runner._release_running_agent_state.assert_not_called()

    @pytest.fixture(params=["dm", "group"])
    def legacy_telegram(self, request):
        """A real default-profile store with an inactive, SQL-NULL-origin row."""
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        # conftest isolates both HERMES_HOME and the default DB path per test.
        root = get_hermes_home()
        config = GatewayConfig(multiplex_profiles=False, group_sessions_per_user=True)
        store = SessionStore(root / "sessions", config)
        try:
            event = _make_event("/resume legacy_target")
            event.source.chat_type = request.param
            if request.param == "group":
                event.source.chat_id = "-67890"
            current = store.get_or_create_session(event.source)
            db = store._db
            assert isinstance(db, SessionDB)
            assert db._own_profile_name() == "default"
            db.create_session(
                "legacy_target", "telegram", session_key=current.session_key,
                chat_id=event.source.chat_id, user_id=event.source.user_id,
                chat_type=event.source.chat_type, thread_id=None,
                origin_json=None, profile_name="default",
            )
            db.set_session_title("legacy_target", "Historical Telegram")
            db.append_message("legacy_target", "user", "Historical transcript")
            db.end_session("legacy_target", "reset")
            runner = _make_runner(session_db=db, event=event, persist_event_origin=False)
            runner.config = config
            runner.session_store = store
            self._guard_side_effects(runner)
            assert runner._gateway_session_origin_for_id("legacy_target") is None
            yield db, runner, event, current
        finally:
            store.close_all_db_handles()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("target", ["legacy_target", "Historical Telegram", "1"])
    async def test_legacy_telegram_real_store_switch_readback(self, legacy_telegram, target):
        from gateway.session import AsyncSessionStore

        db, runner, event, current = legacy_telegram
        key, previous_id = current.session_key, current.session_id
        assert db.get_session("legacy_target")["origin_json"] is None
        assert isinstance(runner.async_session_store, AsyncSessionStore)
        result = await runner._handle_resume_command(
            MessageEvent(text=f"/resume {target}", source=event.source))
        assert "Resumed" in result
        assert runner.session_store.lookup_by_session_key(key).session_id == "legacy_target"
        assert (await runner.async_session_store.get_or_create_session(event.source)).session_id == "legacy_target"
        assert db.get_session(previous_id)["end_reason"] == "session_switch"
        assert db.get_session("legacy_target")["ended_at"] is None
        transcript = await runner.async_session_store.load_transcript("legacy_target")
        assert [m["content"] for m in transcript if m["role"] == "user"] == ["Historical transcript"]
        runner._clear_conversation_scope.assert_called_once_with(key, reason="resume")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("thread", [None, ""])
    async def test_legacy_telegram_listing_is_read_only(self, legacy_telegram, thread):
        db, runner, event, current = legacy_telegram
        db._conn.execute("UPDATE sessions SET thread_id=? WHERE id='legacy_target'", (thread,))
        db._conn.commit()
        before = db.get_session("legacy_target")
        for command in ("/resume", "/sessions", "/sessions full", "/sessions --full"):
            handler = runner._handle_sessions_command if command.startswith("/sessions") else runner._handle_resume_command
            result = await handler(MessageEvent(text=command, source=event.source))
            assert "Historical Telegram" in result
        assert db.get_session("legacy_target") == before
        assert runner.session_store.lookup_by_session_key(current.session_key).session_id == current.session_id
        runner._clear_conversation_scope.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field,value", [
        ("origin_json", ""), ("origin_json", " "), ("origin_json", "null"),
        ("origin_json", "{}"), ("origin_json", "{broken"),
        ("session_key", ""), ("session_key", "agent:work:telegram:dm:67890"),
        ("source", "discord"), ("chat_id", "99999"), ("chat_id", 67890),
        ("user_id", "99999"), ("user_id", 12345), ("user_id", "１２３４５"),
        ("chat_type", "channel"), ("chat_type", None), ("thread_id", "1"),
        ("thread_id", 0), ("profile_name", None), ("profile_name", ""),
        ("profile_name", "work"),
    ])
    async def test_legacy_telegram_rejects_conflicting_rows(self, legacy_telegram, field, value):
        db, runner, event, current = legacy_telegram
        row = {**db.get_session("legacy_target"), field: value}
        assert await runner._resume_target_allowed(event.source, "legacy_target", persisted_row=row) is False
        assert await runner._resume_row_visible(event.source, row, False) is False
        assert row[field] == value
        assert runner.session_store.lookup_by_session_key(current.session_key).session_id == current.session_id

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field", [
        "origin_json", "session_key", "source", "chat_id", "user_id",
        "chat_type", "thread_id", "profile_name",
    ])
    async def test_legacy_telegram_missing_fields_are_not_sql_null(self, legacy_telegram, field):
        db, runner, event, _ = legacy_telegram
        row = db.get_session("legacy_target")
        del row[field]
        assert await runner._resume_target_allowed(event.source, "legacy_target", persisted_row=row) is False
        assert field not in row

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field,value", [
        ("platform", Platform.DISCORD), ("chat_type", "channel"),
        ("thread_id", "1"), ("user_id", "99999"), ("user_id_alt", "12345"),
        ("chat_id_alt", "67890"), ("profile", "work"),
        ("scope_id", "workspace"), ("guild_id", "guild"),
        ("prospective_thread_id", "1"), ("parent_chat_id", "67890"),
    ])
    async def test_legacy_telegram_rejects_nonordinary_callers(self, legacy_telegram, field, value):
        db, runner, event, _ = legacy_telegram
        setattr(event.source, field, value)
        assert await runner._resume_target_allowed(event.source, "legacy_target") is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("multiplex", [True, None])
    async def test_legacy_telegram_requires_multiplex_explicitly_off(self, legacy_telegram, multiplex):
        _, runner, event, _ = legacy_telegram
        runner.config.multiplex_profiles = multiplex
        assert await runner._resume_target_allowed(event.source, "legacy_target") is False

    @pytest.mark.asyncio
    async def test_legacy_telegram_shared_group_stays_closed(self, legacy_telegram):
        db, runner, event, _ = legacy_telegram
        event.source.chat_type = "group"
        runner.config.group_sessions_per_user = False
        row = {**db.get_session("legacy_target"), "chat_type": "group",
               "session_key": runner._session_key_for_source(event.source)}
        assert await runner._resume_target_allowed(event.source, "legacy_target", persisted_row=row) is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("store_owner", ["work", None])
    async def test_legacy_telegram_requires_actual_db_default_owner(self, legacy_telegram, store_owner):
        from hermes_state import AsyncSessionDB, SessionDB

        db, runner, event, _ = legacy_telegram
        root = db.db_path.parent
        path = root / "profiles" / "work" / "state.db" if store_owner else root / "unknown" / "state.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        other_db = SessionDB(path)
        try:
            assert other_db._own_profile_name() == store_owner
            runner._session_db = AsyncSessionDB(other_db)
            # Row and bare key both say default; only the actual DB contradicts them.
            row = db.get_session("legacy_target")
            assert row["profile_name"] == "default"
            assert await runner._resume_target_allowed(event.source, "legacy_target", persisted_row=row) is False
        finally:
            other_db.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field,value", [
        ("origin_json", "{broken"), ("user_id", "99999"), ("profile_name", "work"),
    ])
    async def test_legacy_telegram_denial_preserves_real_route_and_cleanup(self, legacy_telegram, field, value):
        db, runner, event, current = legacy_telegram
        db._conn.execute(f"UPDATE sessions SET {field}=? WHERE id='legacy_target'", (value,))
        db._conn.commit()
        before = db.get_session("legacy_target")
        active_before = db.get_session(current.session_id)
        for target in ("legacy_target", "Historical Telegram", "1"):
            result = await runner._handle_resume_command(
                MessageEvent(text=f"/resume {target}", source=event.source))
            assert "Resumed" not in result
        assert runner.session_store.lookup_by_session_key(current.session_key).session_id == current.session_id
        assert db.get_session("legacy_target") == before
        assert db.get_session(current.session_id) == active_before
        runner._release_running_agent_state.assert_not_called()
        runner._clear_conversation_scope.assert_not_called()
        runner._evict_cached_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_legacy_telegram_keeps_live_origin_priority(self, legacy_telegram):
        _, runner, event, current = legacy_telegram
        runner._gateway_session_origin_for_id = lambda sid: _make_event(chat_id="99999").source
        assert await runner._resume_target_allowed(event.source, "legacy_target") is False
        runner._gateway_session_origin_for_id = lambda sid: event.source
        assert await runner._resume_target_allowed(event.source, "legacy_target") is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("foreign_root", [True, False])
    @pytest.mark.parametrize("live", [True, False])
    @pytest.mark.parametrize("platform", [Platform.TELEGRAM, Platform.MATRIX])
    async def test_compression_requires_authorized_root_and_tip(
        self, tmp_path, foreign_root, live, platform
    ):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "chain.db")
        event = _make_event("/resume chain_root", platform=platform)
        foreign = _make_event(platform=platform, chat_id="foreign-room", user_id="victim").source
        sources = {
            "chain_root": foreign if foreign_root else event.source,
            "chain_tip": event.source if foreign_root else foreign,
        }
        self._create(db, "chain_root", sources["chain_root"], "Chain Work")
        db.end_session("chain_root", "compression")
        self._create(db, "chain_tip", sources["chain_tip"], parent="chain_root")
        db.append_message("chain_tip", "user", "PRIVATE_TIP_PREVIEW")
        runner = _make_runner(session_db=db, event=event, persist_event_origin=False)
        runner._gateway_session_origin_for_id = lambda sid: sources.get(sid) if live else None
        self._guard_side_effects(runner)

        for target in ["chain_root", "Chain Work"]:
            result = await runner._handle_resume_command(
                MessageEvent(text=f"/resume {target}", source=event.source)
            )
            self._assert_denied(runner, result)
            assert "PRIVATE_TIP_PREVIEW" not in result

        # Native display rows combine root provenance with tip display fields.
        projected = next(row for row in db.list_sessions_rich() if row["id"] == "chain_tip")
        assert projected["_lineage_root_id"] == "chain_root"
        assert projected["chat_id"] == sources["chain_root"].chat_id
        assert await runner._resume_row_visible(event.source, projected, False) is False
        numbered = await runner._handle_resume_command(MessageEvent(text="/resume 1", source=event.source))
        assert "Resumed" not in numbered
        runner.session_store.switch_session.assert_not_called()
        db.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("admin_override", [False, True])
    async def test_title_resolution_skips_newer_foreign_numbered_variant(self, tmp_path, admin_override):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "titles.db")
        event = _make_event("/resume " + ("--all " if admin_override else "") + "Shared Title")
        self._create(db, "owned_exact", event.source, "Shared Title")
        foreign = _make_event(chat_id="foreign", user_id="victim").source
        self._create(db, "foreign_numbered", foreign, "Shared Title #2")
        runner = _make_runner(session_db=db, event=event, persist_event_origin=False)
        if admin_override:
            runner.config.platforms[Platform.TELEGRAM] = PlatformConfig(extra={"allow_admin_from": [event.source.user_id]})
        result = await runner._handle_resume_command(event)
        assert "Resumed" in result
        assert runner.session_store.switch_session.call_args.args[1] == "owned_exact"
        db.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("platform", [Platform.TELEGRAM, Platform.MATRIX])
    @pytest.mark.parametrize("bad_field,bad_value", [
        ("origin_json", None), ("origin_json", "{malformed SECRET_ORIGIN"),
        ("origin_json", "{}"), ("session_key", "wrong-key"),
        ("source", "discord"), ("user_id", "contradictory-user"),
        ("chat_id", "contradictory-chat"), ("chat_type", "channel"),
        ("thread_id", "contradictory-thread"),
    ])
    async def test_persisted_origin_contradictions_fail_closed(self, tmp_path, platform, bad_field, bad_value):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "origin.db")
        event = _make_event("/resume target", platform=platform)
        self._create(db, "target", event.source, "PRIVATE_TITLE")
        db._conn.execute(f"UPDATE sessions SET {bad_field} = ? WHERE id = ?", (bad_value, "target"))
        db._conn.commit()
        runner = _make_runner(session_db=db, event=event, persist_event_origin=False)
        self._guard_side_effects(runner)
        result = await runner._handle_resume_command(event)
        self._assert_denied(runner, result)
        assert "PRIVATE_TITLE" not in result
        assert "SECRET_ORIGIN" not in result
        assert await runner._resume_row_visible(event.source, db.get_session("target"), False) is False
        db.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("different_chat", [False, True])
    async def test_same_dm_identity_namespace_transition_requires_canonical_chat(self, tmp_path, different_chat):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "feishu.db")
        stored = SessionSource(platform=Platform.FEISHU, chat_id="oc_dm", chat_type="dm", user_id="legacy-user")
        self._create(db, "target", stored)
        caller = SessionSource(platform=Platform.FEISHU, chat_id="other-dm" if different_chat else "oc_dm",
                               chat_type="dm", user_id="new-user", user_id_alt="open-id")
        runner = _make_runner(session_db=db)
        result = await runner._handle_resume_command(MessageEvent(text="/resume target", source=caller))
        if different_chat:
            assert "No session found" in result
            runner.session_store.switch_session.assert_not_called()
        else:
            assert "Resumed" in result
            runner.session_store.switch_session.assert_called_once()
        db.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("compressed", [False, True])
    async def test_historical_matrix_complete_origin_survives_restart(self, tmp_path, compressed):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "matrix.db")
        event = _make_matrix_event("/resume Historical Matrix Work", thread_id="$thread")
        self._create(db, "root", event.source, "Historical Matrix Work")
        tip = "root"
        if compressed:
            db.end_session("root", "compression")
            tip = "tip"
            self._create(db, tip, event.source, parent="root")
        db.append_message(tip, "user", "Historical preview")
        runner = _make_runner(session_db=db, event=event, persist_event_origin=False)
        runner._gateway_session_origin_for_id = lambda sid: None
        result = await runner._handle_resume_command(event)
        assert "Resumed" in result
        assert runner.session_store.switch_session.call_args.args[1] == tip
        listing = await runner._handle_sessions_command(MessageEvent(text="/sessions --full", source=event.source))
        assert "Historical Matrix Work" in listing
        assert "Historical preview" in listing
        db.close()

    @pytest.mark.parametrize("field,value", [
        ("chat_type", "group"), ("profile", "other-profile"), ("scope_id", "other-workspace"),
    ])
    def test_live_origin_requires_exact_canonical_key(self, field, value):
        caller = SessionSource(platform=Platform.SLACK, chat_id="dm-chat", chat_type="dm", user_id="alice")
        origin = SessionSource.from_dict(caller.to_dict())
        setattr(origin, field, value)
        runner = _make_runner()
        runner.config.multiplex_profiles = True
        assert runner._session_key_for_source(caller) != runner._session_key_for_source(origin)
        assert runner._same_origin_chat(caller, origin) is False

    @pytest.mark.parametrize("field,value", [
        ("platform", "unknown-platform"), ("user_id_alt", 123),
        ("profile", []), ("scope_id", {}), ("prospective_thread_id", False),
    ])
    def test_persisted_origin_decoder_rejects_malformed_identity_fields(self, field, value):
        payload = _make_event().source.to_dict()
        payload[field] = value
        assert _make_runner()._decode_persisted_session_source(payload) is None

    def test_live_origins_without_identity_fail_closed(self):
        runner = _make_runner()
        source = SessionSource(platform=Platform.TELEGRAM, chat_id=None, user_id=None, chat_type="dm")
        assert runner._same_origin_chat(source, source) is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("live", [False, True])
    @pytest.mark.parametrize("chat_type,chat_id,user_id", [
        ("dm", None, None), ("group", "room", None), ("group", None, "alice"),
    ])
    async def test_anonymous_origin_cannot_prove_ownership(self, tmp_path, live, chat_type, chat_id, user_id):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "anonymous.db")
        source = SessionSource(platform=Platform.TELEGRAM, chat_type=chat_type, chat_id=chat_id, user_id=user_id)
        self._create(db, "target", source)
        runner = _make_runner(session_db=db)
        runner._gateway_session_origin_for_id = lambda sid: source if live else None
        self._guard_side_effects(runner)
        result = await runner._handle_resume_command(MessageEvent(text="/resume target", source=source))
        self._assert_denied(runner, result)
        db.close()

    def test_title_candidates_preserve_ranking_escape_and_full_provenance(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "candidate.db")
        source = _make_event().source
        for sid, title in [("exact", "A_%"), ("numbered", "A_% #2"), ("decoy", "ABC #2")]:
            self._create(db, sid, source, title)
        rows = db.list_session_title_candidates("A_%")
        assert [row["id"] for row in rows] == ["numbered", "exact"]
        for row in rows:
            assert row["origin_json"] == json.dumps(source.to_dict())
            assert row["session_key"] == build_session_key(source)
        db.close()


    @pytest.mark.asyncio
    async def test_exact_title_skips_foreign_duplicate_before_authorized_candidate(
        self, tmp_path
    ):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        title = "Shared Historical Title"
        db.create_session(
            "foreign_exact",
            "telegram",
            user_id="foreign-user",
            chat_id="foreign-chat",
        )
        db.set_session_title("foreign_exact", title)
        db.create_session(
            "owned_exact",
            "telegram",
            user_id="12345",
            chat_id="67890",
        )
        # Historical databases can contain duplicate exact titles even though
        # the current setter rejects creating them. Reproduce that legacy row
        # shape directly so authorization must inspect every exact candidate.
        assert db._conn is not None
        db._conn.execute("DROP INDEX idx_sessions_title_unique")
        db._conn.execute(
            "UPDATE sessions SET title = ?, started_at = ? WHERE id = ?",
            (title, 200, "owned_exact"),
        )
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (100, "foreign_exact"),
        )
        db._conn.commit()
        db.create_session(
            "current_session_001",
            "telegram",
            user_id="12345",
            chat_id="67890",
        )

        event = _make_event(text=f"/resume {title}")
        runner = _make_runner(
            session_db=db,
            current_session_id="current_session_001",
            event=event,
        )

        result = await runner._handle_resume_command(event)

        assert "Resumed" in result
        switch_call = getattr(runner.session_store.switch_session, "call_args")
        assert switch_call is not None
        assert switch_call[0][1] == "owned_exact"
        db.close()

    @pytest.mark.asyncio
    async def test_persisted_alt_identity_allows_exact_canonical_owner(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "signal-alt.db")
        owner = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="signal-group",
            chat_type="group",
            user_id="+155****1111",
            user_id_alt="owner-uuid",
        )
        db.create_session(
            "signal_owner",
            Platform.SIGNAL.value,
            user_id=owner.user_id,
            chat_id=owner.chat_id,
            chat_type=owner.chat_type,
        )
        _record_gateway_origin(db, "signal_owner", owner)
        runner = _make_runner(session_db=db)
        runner._gateway_session_origin_for_id = lambda session_id: None

        assert await runner._resume_target_allowed(
            owner, "signal_owner", allow_override=False
        ) is True
        attacker = SessionSource.from_dict(owner.to_dict())
        attacker.user_id_alt = "attacker-uuid"
        assert await runner._resume_target_allowed(
            attacker, "signal_owner", allow_override=False
        ) is False
        db.close()

    @pytest.mark.asyncio
    async def test_persisted_whatsapp_alias_flip_uses_canonical_key(
        self, tmp_path, monkeypatch
    ):
        from hermes_state import SessionDB

        mapping_dir = tmp_path / "whatsapp" / "session"
        mapping_dir.mkdir(parents=True)
        (mapping_dir / "lid-mapping-999999999999999.json").write_text(
            json.dumps("15551234567@s.whatsapp.net"),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        db = SessionDB(db_path=tmp_path / "whatsapp-alias.db")
        stored = SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="999999999999999@lid",
            chat_type="dm",
            user_id="999999999999999@lid",
        )
        db.create_session(
            "whatsapp_alias",
            Platform.WHATSAPP.value,
            user_id=stored.user_id,
            chat_id=stored.chat_id,
            chat_type=stored.chat_type,
        )
        _record_gateway_origin(db, "whatsapp_alias", stored)
        caller = SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="15551234567@s.whatsapp.net",
            chat_type="dm",
            user_id="15551234567@s.whatsapp.net",
        )
        runner = _make_runner(session_db=db)
        runner._gateway_session_origin_for_id = lambda session_id: None

        assert build_session_key(stored) == build_session_key(caller)
        assert await runner._resume_target_allowed(
            caller, "whatsapp_alias", allow_override=False
        ) is True
        db.close()

    @pytest.mark.parametrize(
        "missing_key", ["platform", "user_id", "chat_id", "chat_type", "thread_id"]
    )
    def test_persisted_origin_decoder_requires_complete_non_matrix_payload(
        self, missing_key
    ):
        payload = {
            "platform": "telegram",
            "user_id": "12345",
            "chat_id": "67890",
            "chat_type": "group",
            "thread_id": None,
        }
        payload.pop(missing_key)
        runner = _make_runner()
        assert runner._decode_persisted_session_source(payload) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("target", "set_title"),
        [
            ("foreign_session_id", False),
            ("Guessed Foreign Title", True),
        ],
    )
    async def test_nonmatrix_foreign_and_nonexistent_are_indistinguishable(
        self, tmp_path, target, set_title
    ):
        """A guessed foreign title/id gets exactly the nonexistent response."""
        from hermes_state import SessionDB

        foreign_db = SessionDB(db_path=tmp_path / "foreign.db")
        foreign_db.create_session(
            "foreign_session_id",
            "telegram",
            user_id="victim",
            chat_id="victim-chat",
        )
        if set_title:
            foreign_db.set_session_title("foreign_session_id", target)
        foreign_event = _make_event(text=f"/resume {target}")
        foreign_runner = _make_runner(session_db=foreign_db, event=foreign_event)
        foreign_result = await foreign_runner._handle_resume_command(foreign_event)

        empty_db = SessionDB(db_path=tmp_path / "empty.db")
        missing_event = _make_event(text=f"/resume {target}")
        missing_runner = _make_runner(session_db=empty_db, event=missing_event)
        missing_result = await missing_runner._handle_resume_command(missing_event)

        assert foreign_result == missing_result
        assert "No session found" in foreign_result
        foreign_runner.session_store.switch_session.assert_not_called()
        foreign_runner.session_store.load_transcript.assert_not_called()
        empty_db.close()
        foreign_db.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("target", "resolve_title"),
        [
            ("matrix_foreign_id", False),
            ("Guessed Matrix Title", True),
        ],
    )
    async def test_matrix_foreign_and_nonexistent_are_indistinguishable_without_room_metadata(
        self, target, resolve_title
    ):
        """Neither live Matrix room names nor ids are rendered before auth."""
        caller = _make_matrix_event(text=f"/resume {target}")
        runner = _make_runner(event=caller)
        runner._session_db = AsyncMock()
        foreign_origin = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!classified:example.org",
            chat_name="Classified Foreign Room",
            chat_type="group",
            user_id="@victim:example.org",
        )
        runner._gateway_session_origin_for_id = lambda sid: foreign_origin
        if resolve_title:
            runner._session_db.get_session.return_value = None
            runner._session_db.list_session_title_candidates.return_value = [{"id": "matrix_foreign_id"}]
        else:
            runner._session_db.get_session.return_value = {"id": "matrix_foreign_id"}
        runner._session_db.resolve_resume_session_id.return_value = "matrix_foreign_id"

        foreign_result = await runner._handle_resume_command(caller)

        runner._gateway_session_origin_for_id = lambda session_id: None
        runner._session_db.get_session.return_value = None
        runner._session_db.list_session_title_candidates.return_value = []
        nonexistent_result = await runner._handle_resume_command(caller)

        assert foreign_result == nonexistent_result
        assert "No session found" in foreign_result
        assert "Classified Foreign Room" not in foreign_result
        assert "!classified:example.org" not in foreign_result
        runner.session_store.switch_session.assert_not_called()
        runner.session_store.load_transcript.assert_not_called()

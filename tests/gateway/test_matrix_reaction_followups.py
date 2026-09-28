"""A Matrix reply reaction can resume its owning session once."""

import pytest

from plugins.platforms.matrix.reaction_followups import ReactionWatchStore
from contextlib import closing


def test_queued_nonstreamed_final_arms_after_processing_hook(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from gateway.config import Platform
    from gateway.platforms.base import BasePlatformAdapter, ProcessingOutcome, SendResult
    from gateway.session import SessionSource
    from gateway.platforms.event import MessageEvent
    from plugins.platforms.matrix.adapter import MatrixAdapter
    from plugins.platforms.matrix.adapter_feedback import ReadReceiptMode

    async def exercise():
        adapter = object.__new__(MatrixAdapter)
        adapter._store_dir = tmp_path / "store"
        adapter._reaction_followup_actions = {}
        adapter._active_sessions = {"session": asyncio.Event()}
        adapter._reactions_enabled = False
        adapter._read_receipts_mode = ReadReceiptMode.DISABLED
        adapter._event_session_key = lambda _event: "session"
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:test",
            user_id="@alice:test",
            profile="work",
        )
        event = MessageEvent(text="Queued input", source=source, message_id=None)
        assert await adapter.configure_reaction_followups(
            "session", True, ("👍",), room_id="!room:test",
            requester="@alice:test", thread_id="", profile="work", session_id="sid",
        )

        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        monkeypatch.setattr(
            BasePlatformAdapter,
            "send_final_ledgered",
            AsyncMock(return_value=(SendResult(success=True, message_id="$queued"), adapter)),
        )
        await adapter.send_final_ledgered(event, "session", "Queued answer", {}, reply_to=None)

        store = ReactionWatchStore(tmp_path / "reaction-followups.sqlite")
        assert store.candidate("!room:test", "$queued") is not None
        assert adapter._reaction_followup_actions == {}

    asyncio.run(exercise())


def test_queued_final_watches_terminal_turn_requester_and_thread(tmp_path, monkeypatch):
    import asyncio
    import importlib
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from gateway.config import Platform
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.session import SessionSource
    from gateway.session_context import clear_session_vars, set_session_vars
    from plugins.platforms.matrix.adapter import MatrixAdapter
    from tools.registry import registry

    importlib.import_module("tools.matrix_followup_tool")

    async def exercise():
        adapter = object.__new__(MatrixAdapter)
        adapter._store_dir = tmp_path / "store"
        adapter._reaction_followup_actions = {}
        adapter._active_sessions = {"session": asyncio.Event()}
        outer_source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:test",
            chat_type="group",
            user_id="@alice:test",
            thread_id="$outer",
            profile="work",
        )
        tokens = set_session_vars(
            platform="matrix", chat_id="!room:test", chat_type="group",
            user_id="@bob:test", thread_id="$inner", profile="work",
            session_key="session", session_id="sid", transport_adapter=adapter,
        )
        try:
            configured = json.loads(await asyncio.to_thread(
                registry.dispatch, "matrix_followup", {"enabled": True},
            ))
        finally:
            clear_session_vars(tokens)
        assert configured == {"success": True, "enabled": True, "emoji": []}

        monkeypatch.setattr(
            BasePlatformAdapter,
            "send_final_ledgered",
            AsyncMock(return_value=(SendResult(success=True, message_id="$answer"), adapter)),
        )
        await adapter.send_final_ledgered(
            SimpleNamespace(source=outer_source), "session", "Bob's answer", {}, reply_to=None,
        )

        store = ReactionWatchStore(tmp_path / "reaction-followups.sqlite")
        assert store.candidate("!room:test", "$answer") == {
            "profile": "work",
            "thread_id": "$inner",
            "session_key": "session",
            "session_id": "sid",
            "requester": "@bob:test",
            "delivery_event_id": "$answer",
            "source": {
                **outer_source.to_dict(),
                "user_id": "@bob:test",
                "thread_id": "$inner",
            },
        }

    asyncio.run(exercise())


def test_split_final_delivery_arms_only_successful_replies(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from gateway.config import Platform
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.session import SessionSource
    from plugins.platforms.matrix.adapter import MatrixAdapter

    async def exercise():
        adapter = object.__new__(MatrixAdapter)
        adapter._store_dir = tmp_path / "store"
        adapter._reaction_followup_actions = {}
        adapter._active_sessions = {"session": asyncio.Event()}
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:test",
            user_id="@alice:test",
            thread_id="$thread",
            profile="work",
        )
        event = SimpleNamespace(source=source)
        choice_context = {
            "room_id": "!room:test", "requester": "@alice:test",
            "thread_id": "$thread", "profile": "work", "session_id": "sid",
        }
        await adapter.configure_reaction_followups("session", True, ("👍",), **choice_context)
        await adapter.configure_reaction_followups("session", False, (), **choice_context)
        monkeypatch.setattr(
            BasePlatformAdapter,
            "send_final_ledgered",
            AsyncMock(
                return_value=(SendResult(success=True, message_id="$disabled"), adapter)
            ),
        )
        await adapter.send_final_ledgered(
            event, "session", "No watch", {}, reply_to=None
        )
        store = ReactionWatchStore(tmp_path / "reaction-followups.sqlite")
        assert store.candidate("!room:test", "$disabled") is None

        await adapter.configure_reaction_followups("session", True, ("👍",), **choice_context)
        monkeypatch.setattr(
            BasePlatformAdapter,
            "send_final_ledgered",
            AsyncMock(
                return_value=(
                    SendResult(
                        success=True,
                        message_id="$second",
                        continuation_message_ids=("$first",),
                    ),
                    adapter,
                )
            ),
        )
        await adapter.send_final_ledgered(
            event, "session", "A reply", {}, reply_to=None
        )
        assert (
            store.claim("work", "!room:test", "$first", "@alice:test", "👍",
                        verified_delivery_event_id="$second") is not None
        )
        assert store.claim("work", "!room:test", "$second", "@alice:test", "👍",
                           verified_delivery_event_id="$second") is None

        await adapter.configure_reaction_followups("session", True, (), **choice_context)
        monkeypatch.setattr(
            BasePlatformAdapter,
            "send_final_ledgered",
            AsyncMock(
                return_value=(SendResult(success=False, error="refused"), adapter)
            ),
        )
        await adapter.send_final_ledgered(
            event, "session", "Failed reply", {}, reply_to=None
        )
        assert store.candidate("!room:test", "$failed") is None

        monkeypatch.setattr(
            BasePlatformAdapter,
            "send_final_ledgered",
            AsyncMock(
                return_value=(SendResult(success=True, message_id="$empty"), adapter)
            ),
        )
        await adapter.send_final_ledgered(event, "session", "", {}, reply_to=None)
        assert store.candidate("!room:test", "$empty") is None

    asyncio.run(exercise())


def test_watch_claim_is_scoped_atomic_and_expires_without_sleep(tmp_path):
    now = [1000.0]
    path = tmp_path / "watches.sqlite"
    store = ReactionWatchStore(path, clock=lambda: now[0])
    source = {
        "platform": "matrix",
        "chat_id": "!room:test",
        "thread_id": "$thread",
        "user_id": "@alice:test",
        "profile": "work",
    }
    store.arm(
        "turn-1",
        ("$first", "$second"),
        profile="work",
        room_id="!room:test",
        thread_id="$thread",
        session_key="agent:work:matrix:thread:$thread",
        session_id="sid",
        requester="@alice:test",
        source=source,
        emoji_filter=("👍",),
        delivery_event_id="$second",
    )
    assert store.claim("work", "!room:test", "$first", "@bob:test", "👍",
                       verified_delivery_event_id="$second") is None
    assert store.claim("other", "!room:test", "$first", "@alice:test", "👍",
                       verified_delivery_event_id="$second") is None
    assert store.claim("work", "!other:test", "$first", "@alice:test", "👍",
                       verified_delivery_event_id="$second") is None
    assert store.claim("work", "!room:test", "$first", "@alice:test", "👎",
                       verified_delivery_event_id="$second") is None
    assert (
        store.claim(
            "work", "!room:test", "$first", "@alice:test", "👍",
            verified_delivery_event_id="$preview",
        )
        is None
    )

    restarted = ReactionWatchStore(path, clock=lambda: now[0])
    claimed = restarted.claim("work", "!room:test", "$second", "@alice:test", "👍",
                              verified_delivery_event_id="$second")
    assert claimed == {
        "profile": "work",
        "room_id": "!room:test",
        "thread_id": "$thread",
        "session_key": "agent:work:matrix:thread:$thread",
        "session_id": "sid",
        "requester": "@alice:test",
        "source": source,
        "emoji": "👍",
        "target_event_id": "$second",
        "text_content": "",
    }
    assert restarted.claim("work", "!room:test", "$first", "@alice:test", "👍",
                           verified_delivery_event_id="$second") is None

    store.arm(
        "turn-2",
        ("$later",),
        profile="work",
        room_id="!room:test",
        thread_id="$thread",
        session_key="agent:work:matrix:thread:$thread",
        session_id="sid",
        requester="@alice:test",
        source=source,
        emoji_filter=(),
        delivery_event_id="$later",
    )
    now[0] += 601
    assert restarted.claim("work", "!room:test", "$later", "@alice:test", "✅",
                           verified_delivery_event_id="$later") is None


@pytest.mark.parametrize("bound_session", [False, True])
def test_existing_watch_database_discards_rows_without_delivery_event(tmp_path, bound_session):
    from contextlib import closing
    import json
    import sqlite3

    path = tmp_path / "watches.sqlite"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("""
            CREATE TABLE watches (
                event_id TEXT PRIMARY KEY, turn_id TEXT NOT NULL,
                profile TEXT NOT NULL, room_id TEXT NOT NULL,
                thread_id TEXT NOT NULL, session_key TEXT NOT NULL,
                requester TEXT NOT NULL, source_json TEXT NOT NULL,
                emoji_json TEXT NOT NULL, expires_at REAL NOT NULL
            )
        """)
        db.execute(
            "INSERT INTO watches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("$old", "turn-old", "work", "!room:test", "", "session",
             "@alice:test", json.dumps({"chat_id": "!room:test"}), "[]", 1600.0),
        )
        if bound_session:
            db.execute("ALTER TABLE watches ADD COLUMN session_id TEXT NOT NULL DEFAULT 'sid'")

    store = ReactionWatchStore(path, clock=lambda: 1000.0)
    assert store.claim("work", "!room:test", "$old", "@alice:test", "👍",
                       verified_delivery_event_id="$old") is None
    assert store.candidate("!room:test", "$old") is None

    store.arm(
        "turn-new", ("$new",), profile="work", room_id="!room:test",
        thread_id="", session_key="session", session_id="sid",
        requester="@alice:test", source={"chat_id": "!room:test"}, emoji_filter=(),
        delivery_event_id="$new",
    )
    claimed = store.claim("work", "!room:test", "$new", "@alice:test", "👍",
                          verified_delivery_event_id="$new")
    assert claimed == {
        "profile": "work",
        "room_id": "!room:test",
        "thread_id": "",
        "session_key": "session",
        "session_id": "sid",
        "requester": "@alice:test",
        "source": {"chat_id": "!room:test"},
        "emoji": "👍",
        "target_event_id": "$new",
        "text_content": "",
    }


def test_reaction_intake_starts_one_turn_with_actor_target_and_emoji(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from gateway.config import Platform
    from gateway.session import SessionSource
    from plugins.platforms.matrix.adapter import MatrixAdapter
    from plugins.platforms.matrix.reply_context import (
        MatrixEventContext,
        MatrixEventContextCache,
    )

    async def exercise():
        adapter = object.__new__(MatrixAdapter)
        adapter._store_dir = tmp_path / "store"
        adapter._reaction_watch_store = None
        adapter._ignored_user_patterns = []
        adapter.set_authorization_check(
            lambda user, _chat_type, _chat_id, **_kwargs: user == "@alice:test"
        )
        adapter._is_system_or_bridge_sender = lambda _: False
        adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
        adapter.build_source = lambda **kwargs: SessionSource(
            platform=Platform.MATRIX, profile="work", **kwargs
        )
        adapter._source_session_key = lambda _: "session"
        adapter._session_store = SimpleNamespace(peek_session_id=lambda _key: "sid")
        adapter.handle_message = AsyncMock()
        adapter._event_context_cache = MatrixEventContextCache()
        adapter._event_context_cache.store(
            "!room:test",
            "$second",
            MatrixEventContext("@hermes:test", "The second chunk explains the answer."),
        )
        adapter._client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(return_value={
            "end": "after-final", "chunk": [{"event_id": "$wrong"}, {"event_id": "$react"}],
        })))
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:test",
            user_id="@alice:test",
            thread_id="$thread",
            profile="work",
        )
        adapter._followup_store().arm(
            "turn",
            ("$first", "$second"),
            profile="work",
            room_id="!room:test",
            thread_id="$thread",
            session_key="session",
            session_id="sid",
            requester="@alice:test",
            source=source.to_dict(),
            emoji_filter=("👍",),
            delivery_event_id="$second",
        )
        await adapter._handle_followup_reaction(
            "!room:test", "$first", "👍", "@bob:test", "$bad"
        )
        await adapter._handle_followup_reaction(
            "!room:test", "$first", "👎", "@alice:test", "$wrong"
        )
        await adapter._handle_followup_reaction(
            "!room:test", "$second", "👍", "@alice:test", "$react"
        )
        await adapter._handle_followup_reaction(
            "!room:test", "$first", "👍", "@alice:test", "$repeat"
        )

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert (
            event.source.user_id,
            event.source.thread_id,
            event.message_id,
            event.reply_to_message_id,
            event.allow_gateway_control,
            event.channel_context,
            event.reply_to_text,
            event.reply_to_is_own_message,
            event.defer_until_idle,
        ) == (
            "@alice:test",
            "$thread",
            "$react",
            "$second",
            False,
            "Matrix reaction by @alice:test: 👍 on reply $second (reaction event $react).",
            "The second chunk explains the answer.",
            True,
            True,
        )

    asyncio.run(exercise())


@pytest.mark.parametrize("grant", ("global", "platform", "pairing", "denied"))
def test_followup_reaction_uses_gateway_source_authorization(tmp_path, monkeypatch, grant):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.pairing import PairingStore
    from gateway.run import GatewayRunner
    from gateway.session import SessionStore
    from plugins.platforms.matrix.adapter import MatrixAdapter

    async def exercise():
        for key in ("GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS",
                    "MATRIX_ALLOWED_USERS", "MATRIX_ALLOW_ALL_USERS"):
            monkeypatch.delenv(key, raising=False)
        if grant == "global":
            monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "@alice:test")
        if grant == "platform":
            monkeypatch.setenv("MATRIX_ALLOW_ALL_USERS", "true")

        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig()
        runner.pairing_store = PairingStore()
        if grant == "pairing":
            runner.pairing_store._approve_user("matrix", "@alice:test", "Alice")
        adapter = MatrixAdapter(PlatformConfig(enabled=True))
        runner.adapters = {Platform.MATRIX: adapter}
        runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
        try:
            adapter.gateway_runner = runner
            adapter.set_session_store(runner.session_store)
            adapter._store_dir = tmp_path / "store"
            adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
            adapter.handle_message = AsyncMock()
            adapter._client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(return_value={
                "end": "after-final", "chunk": [{"event_id": "$reaction"}],
            })))
            adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.MATRIX))
            source = adapter.build_source(
                chat_id="!room:test", chat_type="group", user_id="@alice:test",
                thread_id="$thread",
            )
            entry = runner.session_store.get_or_create_session(source)
            assert runner._is_user_authorized_for_source(source) is (grant != "denied")
            assert adapter._is_authorized_user("@alice:test") is (grant != "denied")
            adapter._followup_store().arm(
                "turn", ("$reply",), profile=source.profile or "", room_id="!room:test",
                thread_id="$thread", session_key=entry.session_key, session_id=entry.session_id,
                requester="@alice:test", source=source.to_dict(), emoji_filter=(),
                delivery_event_id="$reply",
            )

            await adapter._handle_followup_reaction(
                "!room:test", "$reply", "👍", "@alice:test", "$reaction",
            )

            if grant == "denied":
                adapter.handle_message.assert_not_awaited()
                assert adapter._followup_store().candidate("!room:test", "$reply") is not None
                return
            adapter.handle_message.assert_awaited_once()
        finally:
            runner.session_store.close_all_db_handles()

    asyncio.run(exercise())


def test_reaction_watch_requires_its_original_conversation_at_claim_and_admission(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixFollowupChoice

    async def exercise():
        current_session = ["old-session"]
        adapter = object.__new__(MatrixAdapter)
        adapter._store_dir = tmp_path / "store"
        adapter._reaction_followup_actions = {
            "session": _MatrixFollowupChoice(
                "turn", (), "!room:test", "@alice:test", "", "work", "old-session",
            )
        }
        adapter._session_store = SimpleNamespace(
            peek_session_id=lambda _key: current_session[0],
        )
        adapter._ignored_user_patterns = []
        adapter.set_authorization_check(
            lambda _user, _chat_type, _chat_id, **_kwargs: True
        )
        adapter._is_system_or_bridge_sender = lambda _user: False
        adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
        adapter.build_source = lambda **kwargs: SessionSource(
            platform=Platform.MATRIX, profile="work", **kwargs,
        )
        adapter._source_session_key = lambda _source: "session"
        adapter.handle_message = AsyncMock()
        adapter._event_context_cache = SimpleNamespace(resolve=AsyncMock(return_value=None))
        adapter._client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=[
            {"end": "after-final"}, {"chunk": [{"event_id": "$queued"}]},
        ])))
        source = SessionSource(
            platform=Platform.MATRIX, chat_id="!room:test",
            user_id="@alice:test", profile="work",
        )
        adapter.on_streamed_final_delivery(source, "session", ("$reply",), "Answer")

        current_session[0] = "new-session"
        await adapter._handle_followup_reaction(
            "!room:test", "$reply", "👍", "@alice:test", "$stale",
        )
        adapter.handle_message.assert_not_awaited()

        current_session[0] = "old-session"
        await adapter._handle_followup_reaction(
            "!room:test", "$reply", "👍", "@alice:test", "$queued",
        )
        event = adapter.handle_message.await_args.args[0]
        assert event.metadata == {
            "gateway_session_key": "session",
            "gateway_session_id": "old-session",
            "gateway_session_strict": True,
        }

        current_session[0] = "new-session"
        runner = object.__new__(GatewayRunner)
        runner._session_key_for_source = lambda _source: "session"
        runner.session_store = SimpleNamespace(
            lookup_by_session_key=lambda _key: SimpleNamespace(session_id=current_session[0]),
        )
        assert await runner._hmwa_resolve_session(event, event.source) is None

    asyncio.run(exercise())


def test_strict_reaction_followup_cannot_queue_into_replacement_turn(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionStore
    from plugins.platforms.matrix.adapter import MatrixAdapter

    async def exercise():
        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "@alice:test")
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@alice:test")
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig()
        runner._busy_input_mode = "queue"
        runner._draining = False
        runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
        try:
            adapter = MatrixAdapter(PlatformConfig(enabled=True))
            runner.adapters = {Platform.MATRIX: adapter}
            adapter.gateway_runner = runner
            adapter.set_session_store(runner.session_store)
            adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.MATRIX))
            adapter.set_message_handler(AsyncMock())
            adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
            adapter._store_dir = tmp_path / "store"
            adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
            adapter._client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(return_value={
                "end": "after-final", "chunk": [{"event_id": "$reaction"}],
            })))
            source = adapter.build_source(
                chat_id="!room:test", chat_type="group", user_id="@alice:test",
            )
            entry = runner.session_store.get_or_create_session(source)
            adapter._followup_store().arm(
                "turn", ("$reply",), profile=source.profile or "", room_id=source.chat_id,
                thread_id="", session_key=entry.session_key, session_id=entry.session_id,
                requester=source.user_id, source=source.to_dict(), emoji_filter=(),
                delivery_event_id="$reply",
            )
            resolving = asyncio.Event()
            resume = asyncio.Event()

            async def resolve(*_args):
                resolving.set()
                await resume.wait()
                return None

            adapter._event_context_cache.resolve = resolve
            intake = asyncio.create_task(adapter._handle_followup_reaction(
                source.chat_id, "$reply", "👍", source.user_id, "$reaction",
            ))
            try:
                await asyncio.wait_for(resolving.wait(), timeout=5)
                replacement = runner.session_store.reset_session(entry.session_key)
                assert replacement.session_id != entry.session_id
                adapter._active_sessions[entry.session_key] = asyncio.Event()
                resume.set()
                await asyncio.wait_for(intake, timeout=5)
            finally:
                resume.set()
                if not intake.done():
                    intake.cancel()
                await asyncio.gather(intake, return_exceptions=True)

            assert adapter._pending_messages == {}
            adapter._message_handler.assert_not_awaited()

            runner._hm_busy_slash_or_photo = AsyncMock(return_value=(False, None))
            runner._queue_or_replace_pending_event = Mock()
            event = MessageEvent(
                text="Reaction by Alice", source=source, defer_until_idle=True,
                allow_gateway_control=False,
                metadata={
                    "gateway_session_key": entry.session_key,
                    "gateway_session_id": entry.session_id,
                    "gateway_session_strict": True,
                },
            )
            assert await runner._hm_handle_running_session_message(
                event, source, entry.session_key,
            ) is None
            runner._queue_or_replace_pending_event.assert_not_called()
        finally:
            runner.session_store.close_all_db_handles()

    asyncio.run(exercise())


def test_encrypted_streamed_reply_keeps_final_text_after_restart(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from gateway.config import Platform, PlatformConfig
    from gateway.session import SessionSource
    from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixFollowupChoice
    from plugins.platforms.matrix.reply_context import MatrixEventContextCache

    async def exercise():
        source = SessionSource(
            platform=Platform.MATRIX, chat_id="!room:test", chat_type="group",
            user_id="@alice:test", profile="work",
        )
        sender = object.__new__(MatrixAdapter)
        sender._store_dir = tmp_path / "store"
        sender.config = PlatformConfig()
        sender.platform = Platform.MATRIX
        sender.set_owner_profile("work")
        session_key = sender._source_session_key(source)
        sender._reaction_followup_actions = {
            session_key: _MatrixFollowupChoice(
                "turn", (), "!room:test", "@alice:test", "", "work", "sid",
            ),
        }
        sender.on_streamed_final_delivery(
            source, session_key, ("$preview",), "Final answer",
        )

        encrypted_edit = SimpleNamespace(
            type="m.room.encrypted", content={"ciphertext": "encrypted edit"},
        )
        encrypted_preview = SimpleNamespace(
            type="m.room.encrypted", sender="@hermes:test",
            unsigned={"m.relations": {"m.replace": encrypted_edit}},
            content={"ciphertext": "encrypted preview"},
        )
        client = SimpleNamespace(
            api=SimpleNamespace(request=AsyncMock(side_effect=[
                {"end": "after-final"}, {"chunk": [{"event_id": "$reaction"}]},
            ])),
            get_event=AsyncMock(return_value=encrypted_preview),
            crypto=SimpleNamespace(decrypt_megolm_event=AsyncMock(
                return_value=SimpleNamespace(
                    type="m.room.message", sender="@hermes:test",
                    unsigned=encrypted_preview.unsigned,
                    content={"msgtype": "m.text", "body": "Draft answer"},
                ),
            )),
        )
        restarted = object.__new__(MatrixAdapter)
        restarted._store_dir = sender._store_dir
        restarted._reaction_watch_store = None
        restarted._ignored_user_patterns = []
        restarted._is_allowed_matrix_room_event = AsyncMock(return_value=True)
        restarted.set_authorization_check(
            lambda _user, _chat_type, _chat_id, **_kwargs: True
        )
        restarted.platform = Platform.MATRIX
        restarted.config = sender.config
        restarted.set_owner_profile("work")
        restarted._session_store = SimpleNamespace(peek_session_id=lambda _key: "sid")
        restarted._event_context_cache = MatrixEventContextCache()
        restarted._client = client
        restarted.handle_message = AsyncMock()

        await restarted._handle_followup_reaction(
            "!room:test", "$preview", "👍", "@alice:test", "$reaction",
        )

        restarted.handle_message.assert_awaited_once()
        delivered = restarted.handle_message.await_args
        assert delivered is not None
        assert delivered.args[0].reply_to_text == "Final answer"

    asyncio.run(exercise())


def test_strict_reaction_followup_is_discarded_before_recursive_drain():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext

    async def exercise():
        source = SessionSource(
            platform=Platform.MATRIX, chat_id="!room:test", user_id="@alice:test",
        )
        event = MessageEvent(
            text="Reaction by Alice", source=source, defer_until_idle=True,
            metadata={
                "gateway_session_key": "session",
                "gateway_session_id": "old-session",
                "gateway_session_strict": True,
            },
        )
        runner = object.__new__(GatewayRunner)
        runner._draining = False
        runner._session_key_for_source = lambda _source: "session"
        current_session = ["new-session"]
        runner.session_store = SimpleNamespace(
            lookup_by_session_key=lambda _key: SimpleNamespace(session_id=current_session[0]),
        )
        overflow = []
        runner._peek_session_state = lambda _key: SimpleNamespace(
            conversation=SimpleNamespace(queued_events=overflow),
        )
        runner._pending_event_audio_paths = Mock(return_value=[])
        pending_messages = {"session": event}
        adapter = SimpleNamespace(
            _pending_messages=pending_messages,
            get_pending_message=lambda key: pending_messages.pop(key, None),
        )

        assert await runner._run_agent_drain_pending(
            {"final_response": "Current answer"}, adapter, source, "session",
        ) == (None, None)

        human = MessageEvent(text="Human follow-up", source=source)
        pending_messages["session"] = event
        overflow.append(human)
        assert await runner._run_agent_drain_pending(
            {"final_response": "Current answer"}, adapter, source, "session",
        ) == (human, human.text)

        turn = TurnContext(
            source=source, session_id="old-session", session_key="session",
            run_generation=1, _interrupt_depth=0, history=[], _status_thread_metadata={},
        )
        runner._run_agent = AsyncMock()
        runner._delivery_adapter_for = lambda _source: None
        runner._intake_adapter_for = lambda _source: None
        runner._is_goal_continuation_event = lambda _event: False
        runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value=event.text)
        runner._reply_anchor_for_event = lambda _event: None
        runner._pinned_channel_inputs = lambda _key, prompt, source, **_kwargs: (prompt, source)
        runner._run_agent_deliver_first_response = AsyncMock()
        runner._persist_prompt_pins = AsyncMock()

        async def reset_during_refresh(*_args):
            current_session[0] = "new-session"

        runner._refresh_agent_cache_message_count = reset_during_refresh
        current_session[0] = "old-session"
        result = {"final_response": "Current answer", "already_sent": True}
        assert await runner._run_agent_queued_followup(
            turn, adapter, event.text, event, result, result, None,
        ) is result
        runner._run_agent.assert_not_awaited()

    asyncio.run(exercise())


def test_streamed_final_arms_visible_original_and_split_events(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixFollowupChoice

    async def exercise():
        adapter = object.__new__(MatrixAdapter)
        adapter._store_dir = tmp_path / "store"
        adapter._reaction_followup_actions = {
            "session": _MatrixFollowupChoice(
                "turn", ("👍",), "!room:test", "@alice:test", "", "work", "sid",
            )
        }
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:test",
            user_id="@alice:test",
            profile="work",
        )
        consumer = SimpleNamespace(
            final_content_delivered=True,
            final_response_sent=True,
            delivered_final_matches=lambda text: text == "Final answer",
            final_message_ids=("$head", "$tail"),
            message_id="$tail",
            adapter=adapter,
        )
        runner = object.__new__(GatewayRunner)
        runner._delivery_adapter_for = lambda _: adapter
        response = {"final_response": "Final answer", "response_previewed": True}
        await runner._run_agent_mark_streamed_delivery(
            response,
            SimpleNamespace(
                stream_consumer_holder=[consumer], source=source, session_key="session"
            ),
        )
        assert response["already_sent"] is True
        store = ReactionWatchStore(tmp_path / "reaction-followups.sqlite")
        assert store.candidate("!room:test", "$head") is not None
        assert store.candidate("!room:test", "$tail") is not None

    asyncio.run(exercise())


def test_transformed_streamed_final_watches_edited_original_event(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from gateway.config import Platform
    from gateway.platforms.base import SendResult
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixFollowupChoice

    async def exercise():
        adapter = object.__new__(MatrixAdapter)
        adapter._store_dir = tmp_path / "store"
        adapter._reaction_followup_actions = {
            "session": _MatrixFollowupChoice(
                "turn", (), "!room:test", "@alice:test", "", "work", "sid",
            )
        }
        adapter.edit_message = AsyncMock(
            return_value=SendResult(success=True, message_id="$replacement")
        )
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:test",
            user_id="@alice:test",
            profile="work",
        )
        consumer = SimpleNamespace(
            final_content_delivered=True,
            final_response_sent=True,
            delivered_final_matches=lambda text: text == "Old answer",
            final_message_ids=("$original",),
            message_id="$original",
            adapter=adapter,
        )
        runner = object.__new__(GatewayRunner)
        response = {"final_response": "New answer", "response_transformed": True}
        await runner._run_agent_mark_streamed_delivery(
            response,
            SimpleNamespace(
                stream_consumer_holder=[consumer], source=source, session_key="session"
            ),
        )

        assert response["already_sent"] is True
        adapter.edit_message.assert_awaited_once()
        store = ReactionWatchStore(tmp_path / "reaction-followups.sqlite")
        assert store.candidate("!room:test", "$original") is not None
        assert store.candidate("!room:test", "$replacement") is None

    asyncio.run(exercise())


def test_streamed_final_ids_exclude_replacement_events_and_prior_segments():
    import asyncio
    from unittest.mock import AsyncMock

    from gateway.platforms.base import SendResult
    from gateway.stream_consumer import GatewayStreamConsumer

    async def exercise():
        consumer = object.__new__(GatewayStreamConsumer)
        consumer.adapter = object()
        consumer._preview_message_ids = {"$prior", "$head", "$tail"}
        consumer._segment_preview_message_ids = {"$head", "$tail"}
        consumer._nonvisible_edit_ids = set()
        consumer._message_id = "$tail"
        consumer._last_sent_text = "Draft"
        consumer._adapter_requires_finalize = False
        consumer._adapter_prefers_fresh_final = lambda _: False
        consumer._should_send_fresh_final = lambda: False
        consumer._edit_message = AsyncMock(
            return_value=SendResult(success=True, message_id="$edit")
        )
        consumer._flood_strikes = 0
        consumer._reopen_seeded_eagerly = False

        assert await consumer._edit_existing(
            "Final answer", finalize=True, is_turn_final=True
        )
        assert consumer.final_message_ids == ("$head", "$tail")

    asyncio.run(exercise())


def test_queued_first_response_arms_streamed_final_before_next_turn(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixFollowupChoice

    async def exercise():
        adapter = object.__new__(MatrixAdapter)
        adapter._store_dir = tmp_path / "store"
        adapter._reaction_followup_actions = {
            "session": _MatrixFollowupChoice(
                "turn", (), "!room:test", "@alice:test", "", "work", "sid",
            )
        }
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:test",
            user_id="@alice:test",
            profile="work",
        )
        consumer = SimpleNamespace(
            final_response_sent=True,
            delivered_final_matches=lambda text: text == "First answer",
            final_message_ids=("$first",),
        )
        runner = object.__new__(GatewayRunner)
        runner._run_agent_stream_confirmed_final_delivery = lambda *_args, **_kwargs: (
            True
        )
        runner._is_intentional_silence = lambda *_args: False
        runner._deliver_queued_first_response = AsyncMock(return_value=True)
        runner._pop_post_delivery_callback = lambda *_args: None
        result = {"final_response": "First answer"}
        turn_ctx = SimpleNamespace(
            mute_notification_reply=False,
            session_key="session",
            stream_consumer_holder=[consumer],
            source=source,
            _status_thread_metadata={},
            event_message_id="$inbound",
            inbound_message_id="$inbound",
            persist_user_display_kind=None,
            reply_expected=True,
            run_generation=1,
        )
        await runner._run_agent_deliver_first_response(
            turn_ctx, adapter, result, result, None
        )
        assert (
            ReactionWatchStore(tmp_path / "reaction-followups.sqlite").candidate(
                "!room:test", "$first"
            )
            is not None
        )

    asyncio.run(exercise())


def test_queued_reconciled_final_arms_edited_reply(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from gateway.config import Platform
    from gateway.platforms.base import SendResult
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixFollowupChoice

    async def exercise():
        adapter = object.__new__(MatrixAdapter)
        adapter._store_dir = tmp_path / "store"
        adapter._reaction_followup_actions = {
            "session": _MatrixFollowupChoice(
                "turn", (), "!room:test", "@alice:test", "", "work", "sid",
            )
        }
        adapter.edit_message = AsyncMock(return_value=SendResult(success=True))
        source = SessionSource(
            platform=Platform.MATRIX, chat_id="!room:test",
            user_id="@alice:test", profile="work",
        )
        consumer = SimpleNamespace(message_id="$preview", final_message_ids=("$preview",))
        runner = object.__new__(GatewayRunner)
        runner._run_agent_stream_confirmed_final_delivery = lambda *_args, **_kwargs: False
        runner._is_intentional_silence = lambda *_args: False
        runner._pop_post_delivery_callback = lambda *_args: None
        runner._deliver_media_from_response = AsyncMock()
        result = {"final_response": "Corrected answer"}
        turn_ctx = SimpleNamespace(
            mute_notification_reply=False, session_key="session",
            stream_consumer_holder=[consumer], source=source,
            _status_thread_metadata={}, event_message_id="$inbound",
            inbound_message_id="$inbound", persist_user_display_kind=None,
            reply_expected=True, run_generation=1,
        )

        await runner._run_agent_deliver_first_response(
            turn_ctx, adapter, result, result, None,
        )

        adapter.edit_message.assert_awaited_once()
        assert result["already_sent"] is True
        assert ReactionWatchStore(tmp_path / "reaction-followups.sqlite").candidate(
            "!room:test", "$preview",
        ) is not None

    asyncio.run(exercise())


def test_split_fallback_final_keeps_every_successful_event_id():
    import asyncio
    from unittest.mock import AsyncMock

    from gateway.platforms.base import SendResult
    from gateway.stream_consumer import GatewayStreamConsumer

    async def exercise():
        consumer = GatewayStreamConsumer(object(), "!room:test")
        consumer._message_id = "$head"
        consumer._preview_message_ids.add("$head")
        consumer._segment_preview_message_ids.add("$head")
        consumer._last_sent_text = "Head "
        consumer._fallback_len_budget = lambda: (len, 700)
        consumer._split_text_chunks = lambda *_args, **_kwargs: ["one ", "two ", "three"]
        consumer._send_with_flood_retry = AsyncMock(side_effect=[
            SendResult(success=True, message_id="$tail1"),
            SendResult(success=True, message_id="$tail2b",
                       continuation_message_ids=("$tail2a",)),
            SendResult(success=True, message_id="$tail3"),
        ])

        await consumer._send_fallback_final("Head one two three")

        assert consumer.final_message_ids == (
            "$head", "$tail1", "$tail2a", "$tail2b", "$tail3",
        )

    asyncio.run(exercise())


def test_split_matrix_send_reports_every_delivered_event():
    import asyncio
    from collections.abc import Callable
    from unittest.mock import AsyncMock

    from plugins.platforms.matrix.adapter import MatrixAdapter

    class SplitMatrixAdapter(MatrixAdapter):
        @staticmethod
        def truncate_message(
            content: str,
            max_length: int = 4096,
            len_fn: Callable[[str], int] | None = None,
        ) -> list[str]:
            return ["abcd", "ef"]

    async def exercise():
        adapter = object.__new__(SplitMatrixAdapter)
        adapter.max_message_length = 4
        adapter.format_message = lambda text: text
        adapter._build_text_message_content = lambda text: {"body": text}
        adapter._apply_relation_metadata = lambda *_args, **_kwargs: None
        adapter._send_room_message = AsyncMock(side_effect=["$one", "$two"])
        result = await adapter.send("!room:test", "abcdef")
        assert (result.success, result.message_id, result.continuation_message_ids) == (
            True,
            "$two",
            ("$one",),
        )

    asyncio.run(exercise())


def test_approval_reactions_take_precedence_over_followup_watches():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from plugins.platforms.matrix.adapter import MatrixAdapter

    async def exercise():
        adapter = object.__new__(MatrixAdapter)
        adapter._is_self_sender = lambda _: False
        adapter._is_duplicate_event = lambda _: False
        adapter._handle_approval_reaction = AsyncMock(return_value=True)
        adapter._handle_model_picker_reaction = AsyncMock(return_value=False)
        adapter._handle_choice_picker_reaction = AsyncMock(return_value=False)
        adapter._handle_followup_reaction = AsyncMock()
        event = SimpleNamespace(
            sender="@alice:test",
            event_id="$reaction",
            room_id="!room:test",
            content={"m.relates_to": {"event_id": "$prompt", "key": "✅"}},
        )
        await adapter._on_reaction(event)
        adapter._handle_approval_reaction.assert_awaited_once()
        adapter._handle_model_picker_reaction.assert_not_awaited()
        adapter._handle_choice_picker_reaction.assert_not_awaited()
        adapter._handle_followup_reaction.assert_not_awaited()

    asyncio.run(exercise())

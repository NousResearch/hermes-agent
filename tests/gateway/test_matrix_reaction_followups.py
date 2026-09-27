"""A Matrix reply reaction can resume its owning session once."""

from plugins.platforms.matrix.reaction_followups import ReactionWatchStore


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
        await adapter.configure_reaction_followups("session", True, ("👍",))
        await adapter.configure_reaction_followups("session", False, ())
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

        await adapter.configure_reaction_followups("session", True, ("👍",))
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
            store.claim("work", "!room:test", "$first", "@alice:test", "👍") is not None
        )
        assert store.claim("work", "!room:test", "$second", "@alice:test", "👍") is None

        await adapter.configure_reaction_followups("session", True, ())
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
        requester="@alice:test",
        source=source,
        emoji_filter=("👍",),
    )
    assert store.claim("work", "!room:test", "$first", "@bob:test", "👍") is None
    assert store.claim("other", "!room:test", "$first", "@alice:test", "👍") is None
    assert store.claim("work", "!other:test", "$first", "@alice:test", "👍") is None
    assert store.claim("work", "!room:test", "$first", "@alice:test", "👎") is None
    assert (
        store.claim(
            "work", "!room:test", "$first", "@alice:test", "👍", reaction_time=999.0
        )
        is None
    )

    restarted = ReactionWatchStore(path, clock=lambda: now[0])
    claimed = restarted.claim("work", "!room:test", "$second", "@alice:test", "👍")
    assert claimed == {
        "profile": "work",
        "room_id": "!room:test",
        "thread_id": "$thread",
        "session_key": "agent:work:matrix:thread:$thread",
        "requester": "@alice:test",
        "source": source,
        "emoji": "👍",
        "target_event_id": "$second",
    }
    assert restarted.claim("work", "!room:test", "$first", "@alice:test", "👍") is None

    store.arm(
        "turn-2",
        ("$later",),
        profile="work",
        room_id="!room:test",
        thread_id="$thread",
        session_key="agent:work:matrix:thread:$thread",
        requester="@alice:test",
        source=source,
        emoji_filter=(),
    )
    now[0] += 601
    assert restarted.claim("work", "!room:test", "$later", "@alice:test", "✅") is None


def test_reaction_intake_starts_one_turn_with_actor_target_and_emoji(tmp_path):
    import asyncio
    from unittest.mock import AsyncMock

    from gateway.config import Platform
    from gateway.session import SessionSource
    from plugins.platforms.matrix.adapter import MatrixAdapter

    async def exercise():
        adapter = object.__new__(MatrixAdapter)
        adapter._store_dir = tmp_path / "store"
        adapter._reaction_watch_store = None
        adapter._ignored_user_patterns = []
        adapter._is_authorized_user = lambda user: user == "@alice:test"
        adapter._is_system_or_bridge_sender = lambda _: False
        adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
        adapter.build_source = lambda **kwargs: SessionSource(
            platform=Platform.MATRIX, profile="work", **kwargs
        )
        adapter._source_session_key = lambda _: "session"
        adapter.handle_message = AsyncMock()
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
            requester="@alice:test",
            source=source.to_dict(),
            emoji_filter=("👍",),
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
        ) == (
            "@alice:test",
            "$thread",
            "$react",
            "$second",
            False,
            "Matrix reaction by @alice:test: 👍 on reply $second (reaction event $react).",
        )

    asyncio.run(exercise())


def test_split_matrix_send_reports_every_delivered_event():
    import asyncio
    from unittest.mock import AsyncMock

    from plugins.platforms.matrix.adapter import MatrixAdapter

    async def exercise():
        adapter = object.__new__(MatrixAdapter)
        adapter.max_message_length = 4
        adapter.format_message = lambda text: text
        adapter.truncate_message = lambda text, limit: ("abcd", "ef")
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

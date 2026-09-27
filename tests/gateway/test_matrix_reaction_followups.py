"""A Matrix reply reaction can resume its owning session once."""

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
            requester="@alice:test", thread_id="", profile="work",
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
            session_key="session", transport_adapter=adapter,
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
            "requester": "@bob:test",
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
            "thread_id": "$thread", "profile": "work",
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
            store.claim("work", "!room:test", "$first", "@alice:test", "👍") is not None
        )
        assert store.claim("work", "!room:test", "$second", "@alice:test", "👍") is None

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
    from plugins.platforms.matrix.reply_context import (
        MatrixEventContext,
        MatrixEventContextCache,
    )

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
        adapter._event_context_cache = MatrixEventContextCache()
        adapter._event_context_cache.store(
            "!room:test",
            "$second",
            MatrixEventContext("@hermes:test", "The second chunk explains the answer."),
        )
        adapter._client = None
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
                "turn", ("👍",), "!room:test", "@alice:test", "", "work",
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
                "turn", (), "!room:test", "@alice:test", "", "work",
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
                "turn", (), "!room:test", "@alice:test", "", "work",
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

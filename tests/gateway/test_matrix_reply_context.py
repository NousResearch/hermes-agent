"""Tests for spec-correct Matrix relation handling and reply context.

Covers the Client-Server API Threading and Rich replies modules (Matrix
v1.13): typed ``m.relates_to`` parsing, reply-fallback stripping and
capture, event-text resolution (cache then API), thread continuation
semantics (``is_falling_back``), and thread-root backfill for sessions
with no history.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from gateway.config import Platform, PlatformConfig, GatewayConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource

from plugins.platforms.matrix.adapter import (
    MatrixAdapter,
    MatrixRelation,
    _extract_mx_reply_quote,
    _parse_relates_to,
    _strip_reply_fallback,
)


def _make_adapter(**extra):
    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
            **extra,
        },
    )
    return MatrixAdapter(config)


# ---------------------------------------------------------------------------
# m.relates_to parsing (spec: Threading / Rich replies modules)
# ---------------------------------------------------------------------------

class TestParseRelatesTo:
    @pytest.mark.parametrize(
        "relates_to, expected",
        [
            # No relation at all.
            ({}, MatrixRelation()),
            (None, MatrixRelation()),
            ("bogus", MatrixRelation()),
            # Rich reply: m.in_reply_to without rel_type.
            (
                {"m.in_reply_to": {"event_id": "$target"}},
                MatrixRelation(reply_target="$target"),
            ),
            # Thread message without any reply metadata.
            (
                {"rel_type": "m.thread", "event_id": "$root"},
                MatrixRelation(thread_root="$root"),
            ),
            # Thread continuation: the in_reply_to pointer is synthetic
            # fallback metadata for unthreaded clients, not a reply.
            (
                {
                    "rel_type": "m.thread",
                    "event_id": "$root",
                    "is_falling_back": True,
                    "m.in_reply_to": {"event_id": "$latest"},
                },
                MatrixRelation(thread_root="$root", thread_fallback_target="$latest"),
            ),
            # Explicit reply within a thread: is_falling_back false.
            (
                {
                    "rel_type": "m.thread",
                    "event_id": "$root",
                    "is_falling_back": False,
                    "m.in_reply_to": {"event_id": "$specific"},
                },
                MatrixRelation(thread_root="$root", reply_target="$specific"),
            ),
            # is_falling_back defaults to false when absent (spec: Replies
            # within threads).
            (
                {
                    "rel_type": "m.thread",
                    "event_id": "$root",
                    "m.in_reply_to": {"event_id": "$specific"},
                },
                MatrixRelation(thread_root="$root", reply_target="$specific"),
            ),
            # Edits.
            (
                {"rel_type": "m.replace", "event_id": "$orig"},
                MatrixRelation(is_edit=True),
            ),
            # Malformed in_reply_to shapes.
            ({"m.in_reply_to": "notadict"}, MatrixRelation()),
            ({"m.in_reply_to": {}}, MatrixRelation()),
            (
                {"rel_type": "m.thread", "event_id": "$root", "m.in_reply_to": {}},
                MatrixRelation(thread_root="$root"),
            ),
            # Other primary relationships can still carry a rich reply.
            (
                {
                    "rel_type": "m.annotation",
                    "event_id": "$other",
                    "m.in_reply_to": {"event_id": "$target"},
                },
                MatrixRelation(reply_target="$target"),
            ),
        ],
    )
    def test_parse(self, relates_to, expected):
        assert _parse_relates_to(relates_to) == expected


# ---------------------------------------------------------------------------
# Reply-fallback stripping (spec: Rich replies, changed in v1.13)
# ---------------------------------------------------------------------------

class TestStripReplyFallback:
    @pytest.mark.parametrize(
        "body, expected",
        [
            # Simple legacy fallback: sender prefix removed from the quote.
            (
                "> <@alice:ex.org> Original message\n\nActual reply",
                ("Actual reply", "Original message"),
            ),
            # Multi-line quote.
            (
                "> <@alice:ex.org> Line 1\n> Line 2\n\nMy response",
                ("My response", "Line 1\nLine 2"),
            ),
            # Bare ">" continuation line (empty quoted line).
            (
                "> <@alice:ex.org> hi\n>\n\nResponse",
                ("Response", "hi"),
            ),
            # No fallback present (v1.13 clients): body passes through.
            ("Just a normal message", ("Just a normal message", None)),
            # Multi-line response after the fallback.
            (
                "> <@alice:ex.org> Original\n\nLine 1\nLine 2\nLine 3",
                ("Line 1\nLine 2\nLine 3", "Original"),
            ),
            # Fallback with no content after it: keep the original body
            # rather than emitting an empty message.
            (
                "> <@alice:ex.org> hi",
                ("> <@alice:ex.org> hi", "hi"),
            ),
        ],
    )
    def test_strip(self, body, expected):
        assert _strip_reply_fallback(body) == expected


class TestExtractMxReplyQuote:
    def test_extracts_quoted_text(self):
        formatted = (
            '<mx-reply><blockquote>'
            '<a href="https://matrix.to/#/!r/$e">In reply to</a> '
            '<a href="https://matrix.to/#/@alice:ex.org">@alice:ex.org</a>'
            '<br/>Original text</blockquote></mx-reply>rest of message'
        )
        assert _extract_mx_reply_quote(formatted) == "Original text"

    def test_strips_nested_tags(self):
        formatted = (
            "<mx-reply><blockquote>"
            '<a href="https://matrix.to/#/!r/$e">In reply to</a> '
            '<a href="https://matrix.to/#/@alice:ex.org">@alice:ex.org</a>'
            "<br/>Some <b>bold</b> text</blockquote></mx-reply>reply"
        )
        assert _extract_mx_reply_quote(formatted) == "Some bold text"

    def test_no_mx_reply_returns_none(self):
        assert _extract_mx_reply_quote("<p>plain formatted body</p>") is None

    def test_mx_reply_not_at_start_returns_none(self):
        # Spec: strip only when formatted_body BEGINS with <mx-reply>.
        formatted = "prefix<mx-reply><blockquote>quoted</blockquote></mx-reply>"
        assert _extract_mx_reply_quote(formatted) is None

    def test_non_string_returns_none(self):
        assert _extract_mx_reply_quote(None) is None
        assert _extract_mx_reply_quote(123) is None


# ---------------------------------------------------------------------------
# Event-text resolution: cache, then API fetch
# ---------------------------------------------------------------------------

class TestResolveEventContext:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._user_id = "@bot:example.org"

    @pytest.mark.asyncio
    async def test_cache_hit_without_client(self):
        """Cached events resolve even when no client is connected."""
        self.adapter._client = None
        self.adapter._cache_event_text("$seen", "@alice:ex.org", "hello there")

        resolved = await self.adapter._resolve_event_context(
            "!room:ex.org", "$seen"
        )

        assert resolved == ("@alice:ex.org", "hello there")

    @pytest.mark.asyncio
    async def test_api_fetch_plain_event(self):
        evt = MagicMock()
        evt.type = "m.room.message"
        evt.sender = "@alice:ex.org"
        evt.content = {"body": "from the api"}
        self.adapter._client = MagicMock()
        self.adapter._client.get_event = AsyncMock(return_value=evt)

        resolved = await self.adapter._resolve_event_context(
            "!room:ex.org", "$remote"
        )

        assert resolved == ("@alice:ex.org", "from the api")

    @pytest.mark.asyncio
    async def test_api_fetch_result_is_cached(self):
        evt = MagicMock()
        evt.type = "m.room.message"
        evt.sender = "@alice:ex.org"
        evt.content = {"body": "fetch once"}
        self.adapter._client = MagicMock()
        self.adapter._client.get_event = AsyncMock(return_value=evt)

        first = await self.adapter._resolve_event_context("!room:ex.org", "$e")
        second = await self.adapter._resolve_event_context("!room:ex.org", "$e")

        assert first == second == ("@alice:ex.org", "fetch once")
        self.adapter._client.get_event.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_api_fetch_strips_legacy_fallback(self):
        """A fetched parent that is itself a legacy reply keeps only its own
        text, not its embedded quote."""
        evt = MagicMock()
        evt.type = "m.room.message"
        evt.sender = "@alice:ex.org"
        evt.content = {"body": "> <@bob:ex.org> earlier\n\nalice's actual text"}
        self.adapter._client = MagicMock()
        self.adapter._client.get_event = AsyncMock(return_value=evt)

        resolved = await self.adapter._resolve_event_context("!room:ex.org", "$e")

        assert resolved == ("@alice:ex.org", "alice's actual text")

    @pytest.mark.asyncio
    async def test_encrypted_event_is_decrypted(self):
        from plugins.platforms.matrix.adapter import EventType

        encrypted = MagicMock()
        encrypted.type = EventType.ROOM_ENCRYPTED
        decrypted = MagicMock()
        decrypted.type = "m.room.message"
        decrypted.sender = "@alice:ex.org"
        decrypted.content = MagicMock(spec=["body"])
        decrypted.content.body = "secret text"

        self.adapter._client = MagicMock()
        self.adapter._client.get_event = AsyncMock(return_value=encrypted)
        self.adapter._client.crypto = MagicMock()
        self.adapter._client.crypto.decrypt_megolm_event = AsyncMock(
            return_value=decrypted
        )

        resolved = await self.adapter._resolve_event_context("!room:ex.org", "$enc")

        assert resolved == ("@alice:ex.org", "secret text")

    @pytest.mark.asyncio
    async def test_decryption_failure_returns_none(self):
        from plugins.platforms.matrix.adapter import EventType

        encrypted = MagicMock()
        encrypted.type = EventType.ROOM_ENCRYPTED
        self.adapter._client = MagicMock()
        self.adapter._client.get_event = AsyncMock(return_value=encrypted)
        self.adapter._client.crypto = MagicMock()
        self.adapter._client.crypto.decrypt_megolm_event = AsyncMock(
            side_effect=Exception("no session")
        )

        resolved = await self.adapter._resolve_event_context("!room:ex.org", "$enc")

        assert resolved is None

    @pytest.mark.asyncio
    async def test_api_error_returns_none(self):
        self.adapter._client = MagicMock()
        self.adapter._client.get_event = AsyncMock(side_effect=Exception("404"))

        resolved = await self.adapter._resolve_event_context("!room:ex.org", "$gone")

        assert resolved is None

    @pytest.mark.asyncio
    async def test_no_client_no_cache_returns_none(self):
        self.adapter._client = None

        resolved = await self.adapter._resolve_event_context("!room:ex.org", "$x")

        assert resolved is None

    def test_cache_is_bounded(self):
        for i in range(600):
            self.adapter._cache_event_text(f"$e{i}", "@a:ex.org", f"m{i}")

        assert self.adapter._cached_event_text("$e0") is None
        assert self.adapter._cached_event_text("$e599") == ("@a:ex.org", "m599")


# ---------------------------------------------------------------------------
# Text handler: reply semantics per is_falling_back
# ---------------------------------------------------------------------------

class TestTextMessageReplySemantics:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._user_id = "@bot:example.org"
        self.adapter._is_dm_room = AsyncMock(return_value=True)
        self.adapter._background_read_receipt = MagicMock()
        self.adapter._text_batch_delay_seconds = 0
        self.adapter._require_mention = True
        self.adapter._free_rooms = set()

        display_names = {
            "@alice:ex.org": "Alice",
            "@bot:example.org": "Hermes",
        }

        async def _display_name(room_id, user_id):
            return display_names.get(user_id, user_id)

        self.adapter._get_display_name = AsyncMock(side_effect=_display_name)
        self.adapter._resolve_event_context = AsyncMock(
            return_value=("@alice:ex.org", "parent text")
        )

    async def _dispatch(self, body, relates_to, formatted_body=None):
        captured = None

        async def capture(msg_event):
            nonlocal captured
            captured = msg_event

        self.adapter.handle_message = capture
        source_content = {"msgtype": "m.text", "body": body}
        if formatted_body is not None:
            source_content["formatted_body"] = formatted_body
        await self.adapter._handle_text_message(
            room_id="!room:ex.org",
            sender="@alice:ex.org",
            event_id="$trigger",
            event_ts=0.0,
            source_content=source_content,
            relates_to=relates_to,
        )
        return captured

    @pytest.mark.asyncio
    async def test_rich_reply_resolves_parent(self):
        event = await self._dispatch(
            "what about this?",
            {"m.in_reply_to": {"event_id": "$parent"}},
        )

        assert event.reply_to_message_id == "$parent"
        assert event.reply_to_text == "parent text"
        assert event.reply_to_author_id == "@alice:ex.org"
        assert event.reply_to_author_name == "Alice"
        assert event.reply_to_is_own_message is False
        self.adapter._resolve_event_context.assert_awaited_once_with(
            "!room:ex.org", "$parent"
        )

    @pytest.mark.asyncio
    async def test_reply_to_own_message_sets_flag(self):
        self.adapter._resolve_event_context = AsyncMock(
            return_value=("@bot:example.org", "earlier bot reply")
        )

        event = await self._dispatch(
            "it's here now, right?",
            {"m.in_reply_to": {"event_id": "$bot_msg"}},
        )

        assert event.reply_to_text == "earlier bot reply"
        assert event.reply_to_author_name == "Hermes"
        assert event.reply_to_is_own_message is True

    @pytest.mark.asyncio
    async def test_thread_continuation_is_not_a_reply(self):
        """is_falling_back=true means the in_reply_to pointer is synthetic
        fallback metadata; the message must not surface as a reply."""
        event = await self._dispatch(
            "continuing the thread",
            {
                "rel_type": "m.thread",
                "event_id": "$root",
                "is_falling_back": True,
                "m.in_reply_to": {"event_id": "$latest"},
            },
        )

        assert event.reply_to_message_id is None
        assert event.reply_to_text is None
        assert event.source.thread_id == "$root"
        self.adapter._resolve_event_context.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_reply_within_thread(self):
        event = await self._dispatch(
            "replying to a specific message",
            {
                "rel_type": "m.thread",
                "event_id": "$root",
                "is_falling_back": False,
                "m.in_reply_to": {"event_id": "$specific"},
            },
        )

        assert event.reply_to_message_id == "$specific"
        assert event.reply_to_text == "parent text"
        assert event.source.thread_id == "$root"

    @pytest.mark.asyncio
    async def test_unresolvable_parent_degrades_gracefully(self):
        self.adapter._resolve_event_context = AsyncMock(return_value=None)

        event = await self._dispatch(
            "reply into the void",
            {"m.in_reply_to": {"event_id": "$gone"}},
        )

        assert event.reply_to_message_id == "$gone"
        assert event.reply_to_text is None
        assert event.reply_to_author_id is None
        assert event.reply_to_is_own_message is False

    @pytest.mark.asyncio
    async def test_legacy_fallback_used_without_api_call(self):
        """A legacy body fallback supplies the quote; no fetch needed."""
        event = await self._dispatch(
            "> <@alice:ex.org> the original\n\nmy reply",
            {"m.in_reply_to": {"event_id": "$parent"}},
        )

        assert event.text == "my reply"
        assert event.reply_to_message_id == "$parent"
        assert event.reply_to_text == "the original"
        self.adapter._resolve_event_context.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_legacy_mx_reply_used_without_api_call(self):
        formatted = (
            "<mx-reply><blockquote>"
            '<a href="https://matrix.to/#/!r/$e">In reply to</a> '
            '<a href="https://matrix.to/#/@alice:ex.org">@alice:ex.org</a>'
            "<br/>html original</blockquote></mx-reply>my reply"
        )
        event = await self._dispatch(
            "my reply",
            {"m.in_reply_to": {"event_id": "$parent"}},
            formatted_body=formatted,
        )

        assert event.reply_to_text == "html original"
        self.adapter._resolve_event_context.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_inbound_message_is_cached(self):
        await self._dispatch("remember me", {})

        assert self.adapter._cached_event_text("$trigger") == (
            "@alice:ex.org",
            "remember me",
        )

    @pytest.mark.asyncio
    async def test_plain_message_has_no_reply_fields(self):
        event = await self._dispatch("no relation at all", {})

        assert event.reply_to_message_id is None
        assert event.reply_to_text is None
        self.adapter._resolve_event_context.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_blockquote_without_reply_is_preserved(self):
        """A '> ' blockquote in a message with no reply relation is content,
        not a fallback — it must not be stripped."""
        event = await self._dispatch("> This is a blockquote", {})

        assert event.text == "> This is a blockquote"


# ---------------------------------------------------------------------------
# Media handler parity
# ---------------------------------------------------------------------------

class TestMediaMessageReplySemantics:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._user_id = "@bot:example.org"
        self.adapter._is_dm_room = AsyncMock(return_value=True)
        self.adapter._background_read_receipt = MagicMock()
        self.adapter._require_mention = True
        self.adapter._free_rooms = set()
        self.adapter._get_display_name = AsyncMock(return_value="Alice")
        self.adapter._resolve_event_context = AsyncMock(
            return_value=("@alice:ex.org", "look at this")
        )

    @pytest.mark.asyncio
    async def test_media_reply_resolves_parent(self):
        captured = None

        async def capture(msg_event):
            nonlocal captured
            captured = msg_event

        self.adapter.handle_message = capture
        await self.adapter._handle_media_message(
            room_id="!room:ex.org",
            sender="@alice:ex.org",
            event_id="$media",
            event_ts=0.0,
            source_content={"msgtype": "m.image", "body": "photo.jpg"},
            relates_to={"m.in_reply_to": {"event_id": "$parent"}},
            msgtype="m.image",
        )

        assert captured is not None
        assert captured.reply_to_message_id == "$parent"
        assert captured.reply_to_text == "look at this"
        assert captured.reply_to_author_name == "Alice"

    @pytest.mark.asyncio
    async def test_media_thread_continuation_is_not_a_reply(self):
        captured = None

        async def capture(msg_event):
            nonlocal captured
            captured = msg_event

        self.adapter.handle_message = capture
        await self.adapter._handle_media_message(
            room_id="!room:ex.org",
            sender="@alice:ex.org",
            event_id="$media",
            event_ts=0.0,
            source_content={"msgtype": "m.image", "body": "photo.jpg"},
            relates_to={
                "rel_type": "m.thread",
                "event_id": "$root",
                "is_falling_back": True,
                "m.in_reply_to": {"event_id": "$latest"},
            },
            msgtype="m.image",
        )

        assert captured is not None
        assert captured.reply_to_message_id is None
        assert captured.source.thread_id == "$root"
        self.adapter._resolve_event_context.assert_not_awaited()


# ---------------------------------------------------------------------------
# Inbound relation normalisation
# ---------------------------------------------------------------------------

class TestInboundRelationNormalisation:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._user_id = "@bot:example.org"
        self.adapter._startup_ts = 0.0
        self.adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
        self.adapter._handle_text_message = AsyncMock()

    def _event(self, relates_to):
        event = MagicMock()
        event.room_id = "!room:ex.org"
        event.sender = "@alice:ex.org"
        event.event_id = "$malformed"
        event.timestamp = 1_800_000_000_000
        event.content = {
            "msgtype": "m.text",
            "body": "hello",
            "m.relates_to": relates_to,
        }
        return event

    @pytest.mark.asyncio
    async def test_non_dict_relates_to_still_dispatches(self):
        """A malformed (non-dict) m.relates_to must not crash the event
        pipeline; the message dispatches with an empty relation."""
        await self.adapter._on_room_message(self._event("bogus"))

        self.adapter._handle_text_message.assert_awaited_once()
        kwargs = self.adapter._handle_text_message.await_args.kwargs
        args = self.adapter._handle_text_message.await_args.args
        relates_to = kwargs.get("relates_to", args[-1] if args else None)
        assert relates_to == {}

    @pytest.mark.asyncio
    async def test_list_relates_to_still_dispatches(self):
        await self.adapter._on_room_message(self._event(["not", "a", "dict"]))

        self.adapter._handle_text_message.assert_awaited_once()


# ---------------------------------------------------------------------------
# Cache maintenance: edits and redactions
# ---------------------------------------------------------------------------

class TestCacheMaintenance:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._user_id = "@bot:example.org"
        self.adapter._cache_event_text("$orig", "@alice:ex.org", "first version")

    def test_edit_updates_cached_text(self):
        self.adapter._apply_edit_to_cache(
            "@alice:ex.org",
            {
                "m.relates_to": {"rel_type": "m.replace", "event_id": "$orig"},
                "m.new_content": {"msgtype": "m.text", "body": "second version"},
                "body": "* second version",
            },
        )

        assert self.adapter._cached_event_text("$orig") == (
            "@alice:ex.org",
            "second version",
        )

    def test_edit_of_unseen_event_is_cached(self):
        """Only the original sender can edit (servers enforce this), so an
        edit is a trustworthy text source even for events we never saw."""
        self.adapter._apply_edit_to_cache(
            "@alice:ex.org",
            {
                "m.relates_to": {"rel_type": "m.replace", "event_id": "$unseen"},
                "m.new_content": {"msgtype": "m.text", "body": "edited text"},
                "body": "* edited text",
            },
        )

        assert self.adapter._cached_event_text("$unseen") == (
            "@alice:ex.org",
            "edited text",
        )

    def test_edit_without_new_content_is_ignored(self):
        self.adapter._apply_edit_to_cache(
            "@alice:ex.org",
            {"m.relates_to": {"rel_type": "m.replace", "event_id": "$orig"}},
        )

        assert self.adapter._cached_event_text("$orig") == (
            "@alice:ex.org",
            "first version",
        )

    @pytest.mark.asyncio
    async def test_redaction_evicts_cached_text(self):
        evt = MagicMock()
        evt.redacts = "$orig"

        await self.adapter._on_redaction(evt)

        assert self.adapter._cached_event_text("$orig") is None

    @pytest.mark.asyncio
    async def test_redaction_of_unknown_event_is_noop(self):
        evt = MagicMock()
        evt.redacts = "$never-seen"

        await self.adapter._on_redaction(evt)

        assert self.adapter._cached_event_text("$orig") == (
            "@alice:ex.org",
            "first version",
        )


# ---------------------------------------------------------------------------
# Outbound send caching
# ---------------------------------------------------------------------------

class TestOutboundEventCaching:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._user_id = "@bot:example.org"
        self.adapter._client = MagicMock()

    @pytest.mark.asyncio
    async def test_sent_message_is_cached_by_event_id(self):
        self.adapter._client.send_message_event = AsyncMock(return_value="$sent1")

        result = await self.adapter.send("!room:ex.org", "hello from the bot")

        assert result.success is True
        assert result.message_id == "$sent1"
        cached = self.adapter._cached_event_text("$sent1")
        assert cached is not None
        assert cached[0] == "@bot:example.org"
        assert "hello from the bot" in cached[1]


# ---------------------------------------------------------------------------
# Thread-root backfill: adapter capability
# ---------------------------------------------------------------------------

def _thread_chunk_event(event_id, sender, body):
    return {
        "event_id": event_id,
        "sender": sender,
        "type": "m.room.message",
        "content": {"msgtype": "m.text", "body": body},
    }


class TestFetchThreadContext:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._user_id = "@bot:example.org"
        self.adapter._client = MagicMock()

        display_names = {
            "@alice:ex.org": "Alice",
            "@bot:example.org": "Hermes",
        }

        async def _display_name(room_id, user_id):
            return display_names.get(user_id, user_id)

        self.adapter._get_display_name = AsyncMock(side_effect=_display_name)
        self.adapter._resolve_event_context = AsyncMock(
            return_value=("@alice:ex.org", "root message")
        )
        # Relations endpoint returns newest-first (dir=b).
        self.adapter._client.api.request = AsyncMock(
            return_value={
                "chunk": [
                    _thread_chunk_event("$e3", "@alice:ex.org", "latest question"),
                    _thread_chunk_event("$e2", "@bot:example.org", "earlier reply"),
                ]
            }
        )

    @pytest.mark.asyncio
    async def test_formats_thread_chronologically_with_root_first(self):
        context = await self.adapter.fetch_thread_context(
            "!room:ex.org", "$root"
        )

        assert context == (
            "[Earlier messages in this thread]\n"
            "[Alice] root message\n"
            "[Hermes] earlier reply\n"
            "[Alice] latest question"
        )

    @pytest.mark.asyncio
    async def test_excludes_triggering_event(self):
        context = await self.adapter.fetch_thread_context(
            "!room:ex.org", "$root", exclude_event_id="$e3"
        )

        assert context == (
            "[Earlier messages in this thread]\n"
            "[Alice] root message\n"
            "[Hermes] earlier reply"
        )

    @pytest.mark.asyncio
    async def test_api_error_returns_none(self):
        self.adapter._client.api.request = AsyncMock(side_effect=Exception("boom"))

        context = await self.adapter.fetch_thread_context("!room:ex.org", "$root")

        assert context is None

    @pytest.mark.asyncio
    async def test_no_client_returns_none(self):
        self.adapter._client = None

        context = await self.adapter.fetch_thread_context("!room:ex.org", "$root")

        assert context is None

    @pytest.mark.asyncio
    async def test_disabled_by_zero_limit(self):
        adapter = _make_adapter(thread_backfill_limit=0)
        adapter._client = MagicMock()
        adapter._client.api.request = AsyncMock()

        context = await adapter.fetch_thread_context("!room:ex.org", "$root")

        assert context is None
        adapter._client.api.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_undecryptable_events_are_skipped(self):
        self.adapter._client.api.request = AsyncMock(
            return_value={
                "chunk": [
                    _thread_chunk_event("$e3", "@alice:ex.org", "readable"),
                    {
                        "event_id": "$enc",
                        "sender": "@alice:ex.org",
                        "type": "m.room.encrypted",
                        "content": {"algorithm": "m.megolm.v1.aes-sha2"},
                    },
                ]
            }
        )
        self.adapter._client.crypto = None

        context = await self.adapter.fetch_thread_context("!room:ex.org", "$root")

        assert context == (
            "[Earlier messages in this thread]\n"
            "[Alice] root message\n"
            "[Alice] readable"
        )

    @pytest.mark.asyncio
    async def test_root_only_thread_still_produces_context(self):
        self.adapter._client.api.request = AsyncMock(return_value={"chunk": []})

        context = await self.adapter.fetch_thread_context("!room:ex.org", "$root")

        assert context == (
            "[Earlier messages in this thread]\n[Alice] root message"
        )

    @pytest.mark.asyncio
    async def test_nothing_resolvable_returns_none(self):
        self.adapter._resolve_event_context = AsyncMock(return_value=None)
        self.adapter._client.api.request = AsyncMock(return_value={"chunk": []})

        context = await self.adapter.fetch_thread_context("!room:ex.org", "$root")

        assert context is None


# ---------------------------------------------------------------------------
# Gateway hook: backfill thread context into history-less sessions
# ---------------------------------------------------------------------------

class TestGatewayThreadBackfill:
    @pytest.fixture()
    def runner(self):
        from gateway.run import GatewayRunner

        r = GatewayRunner.__new__(GatewayRunner)
        r.config = GatewayConfig(group_sessions_per_user=False)
        r.adapters = {}
        r._model = "test-model"
        r._base_url = ""
        r._has_setup_skill = lambda: False
        return r

    @pytest.fixture()
    def source(self):
        return SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:ex.org",
            chat_type="group",
            user_name="iain",
            thread_id="$root",
        )

    def _adapter_with_context(self, context):
        adapter = MagicMock(spec=["fetch_thread_context"])
        adapter.fetch_thread_context = AsyncMock(return_value=context)
        return adapter

    @pytest.mark.asyncio
    async def test_fresh_session_in_thread_gets_backfill(self, runner, source):
        adapter = self._adapter_with_context(
            "[Earlier messages in this thread]\n[Alice] the root"
        )
        runner.adapters = {Platform.MATRIX: adapter}
        event = MessageEvent(text="it's here now, right?", source=source,
                             message_id="$trigger")

        result = await runner._prepare_inbound_message_text(
            event=event, source=source, history=[],
        )

        assert result.startswith("[Earlier messages in this thread]")
        assert "[New message]" in result
        assert "it's here now, right?" in result
        adapter.fetch_thread_context.assert_awaited_once_with(
            "!room:ex.org", "$root", exclude_event_id="$trigger"
        )

    @pytest.mark.asyncio
    async def test_session_with_history_skips_backfill(self, runner, source):
        adapter = self._adapter_with_context("should not appear")
        runner.adapters = {Platform.MATRIX: adapter}
        event = MessageEvent(text="hello", source=source)

        result = await runner._prepare_inbound_message_text(
            event=event, source=source, history=[{"role": "user", "content": "x"}],
        )

        assert "should not appear" not in result
        adapter.fetch_thread_context.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unthreaded_message_skips_backfill(self, runner):
        adapter = self._adapter_with_context("should not appear")
        runner.adapters = {Platform.MATRIX: adapter}
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:ex.org",
            chat_type="group",
            user_name="iain",
        )
        event = MessageEvent(text="hello", source=source)

        result = await runner._prepare_inbound_message_text(
            event=event, source=source, history=[],
        )

        assert "should not appear" not in result
        adapter.fetch_thread_context.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_existing_channel_context_is_preserved(self, runner, source):
        adapter = self._adapter_with_context("should not appear")
        runner.adapters = {Platform.MATRIX: adapter}
        event = MessageEvent(
            text="hello",
            source=source,
            channel_context="[Recent channel messages]\n[Bob] existing",
        )

        result = await runner._prepare_inbound_message_text(
            event=event, source=source, history=[],
        )

        assert result.startswith("[Recent channel messages]")
        assert "should not appear" not in result
        adapter.fetch_thread_context.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_adapter_without_capability_is_fine(self, runner, source):
        runner.adapters = {Platform.MATRIX: object()}
        event = MessageEvent(text="hello", source=source)

        result = await runner._prepare_inbound_message_text(
            event=event, source=source, history=[],
        )

        assert "hello" in result

    @pytest.mark.asyncio
    async def test_fetch_error_degrades_gracefully(self, runner, source):
        adapter = MagicMock(spec=["fetch_thread_context"])
        adapter.fetch_thread_context = AsyncMock(side_effect=Exception("boom"))
        runner.adapters = {Platform.MATRIX: adapter}
        event = MessageEvent(text="still processed", source=source)

        result = await runner._prepare_inbound_message_text(
            event=event, source=source, history=[],
        )

        assert "still processed" in result

    @pytest.mark.asyncio
    async def test_internal_events_skip_backfill(self, runner, source):
        adapter = self._adapter_with_context("should not appear")
        runner.adapters = {Platform.MATRIX: adapter}
        event = MessageEvent(text="synthetic", source=source, internal=True)

        result = await runner._prepare_inbound_message_text(
            event=event, source=source, history=[],
        )

        assert "should not appear" not in result
        adapter.fetch_thread_context.assert_not_awaited()

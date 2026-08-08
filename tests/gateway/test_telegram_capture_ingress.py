"""Slice 1.1R-B: capture-aware pre-ack queue seam.

Real PTB ``Update``/``Message`` fixtures, a real temporary SQLite database
per test (no mocked persistence) -- proving the durable-before-ack guarantee
at the one seam early enough for both polling and webhook: ``Queue.put()``.
"""
import asyncio
import datetime
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

# tests/gateway/conftest.py unconditionally installs a MagicMock stand-in for
# the ``telegram`` package at collection time (see its ``_ensure_telegram_mock``
# docstring) so unrelated adapter tests don't need python-telegram-bot
# installed. This slice's TDD sequence explicitly requires real PTB fixtures
# and a real temporary SQLite database, no mocked persistence -- so force the
# genuine installed package back in before importing anything that touches it.
for _mod_name in [n for n in list(sys.modules) if n == "telegram" or n.startswith("telegram.")]:
    if not hasattr(sys.modules[_mod_name], "__file__"):
        del sys.modules[_mod_name]
import telegram as _real_telegram  # noqa: E402

assert hasattr(_real_telegram, "__file__"), (
    "expected the real python-telegram-bot package, not tests/gateway/conftest.py's mock"
)

# Other Telegram test modules can import capture_ingress during collection while
# conftest's lightweight telegram stand-in is installed. Reload this focused
# module only after restoring real PTB so Update and TelegramError are real
# classes regardless of test collection order.
sys.modules.pop("plugins.platforms.telegram.capture_ingress", None)

from telegram import CallbackQuery, Chat, Message, Update, User  # noqa: E402

from plugins.platforms.telegram.capture_ingress import (
    CaptureAwareQueue,
    CaptureIngressStore,
    CapturePersistenceError,
    DUPLICATE_SAME,
    INSERTED,
    RouteConflict,
    RoutePolicyTable,
    canonicalize_update,
    classify_event_type,
    compute_event_id,
    compute_payload_hash,
    normalize_thread_id,
)

ACCOUNT_ID = 777
PROFILE = "default"
CAPTURE_CHAT_ID = -1001
CAPTURE_THREAD_ID = 271


def _chat(chat_id=CAPTURE_CHAT_ID, is_forum=True, chat_type="supergroup"):
    return Chat(id=chat_id, type=chat_type, is_forum=is_forum)


def _user(user_id=555, is_bot=False):
    return User(id=user_id, first_name="Alice", is_bot=is_bot)


def _message(
    *,
    message_id=42,
    chat=None,
    from_user=None,
    text="hello",
    thread_id=CAPTURE_THREAD_ID,
    **extra,
):
    return Message(
        message_id=message_id,
        date=datetime.datetime.now(datetime.timezone.utc),
        chat=chat or _chat(),
        from_user=from_user if from_user is not None else _user(),
        text=text,
        message_thread_id=thread_id,
        is_topic_message=thread_id is not None,
        **extra,
    )


def _update(update_id=1000, message=None, **kwargs):
    return Update(update_id=update_id, message=message if message is not None else _message(**kwargs))


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "capture_ingress.db"


@pytest.fixture()
def store(db_path):
    s = CaptureIngressStore(db_path)
    yield s
    s.close()


def _capture_only_routes():
    return [
        {
            "chat_id": CAPTURE_CHAT_ID,
            "thread_id": CAPTURE_THREAD_ID,
            "mode": "capture_only",
            "sink": "braindump",
            "policy_version": "1.0.0",
        }
    ]


def _agent_routes():
    return [
        {
            "chat_id": CAPTURE_CHAT_ID,
            "thread_id": CAPTURE_THREAD_ID,
            "mode": "agent",
            "sink": "agent-dispatch",
            "policy_version": "1.0.0",
        }
    ]


def _drop_routes():
    return [
        {
            "chat_id": CAPTURE_CHAT_ID,
            "thread_id": CAPTURE_THREAD_ID,
            "mode": "drop",
            "sink": "n-a",
            "policy_version": "1.0.0",
        }
    ]


def _make_queue(store, routes, *, is_own=False, is_authorized=True, alert=None):
    return CaptureAwareQueue(
        store=store,
        route_table_provider=lambda: RoutePolicyTable(routes),
        account_id_provider=lambda: ACCOUNT_ID,
        profile_provider=lambda: PROFILE,
        thread_id_resolver=lambda message: (
            str(message.message_thread_id) if message.message_thread_id is not None else None
        ),
        is_own_message=lambda message: is_own,
        is_authorized_sender=lambda message: is_authorized,
        alert_failure=(lambda failure, _source: alert(failure)) if alert else None,
    )


def _ledger_row(store, event_id):
    cur = store._conn.execute(
        "SELECT * FROM ingress_ledger WHERE event_id = ?", (event_id,)
    )
    row = cur.fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def _payload_row(store, event_id):
    cur = store._conn.execute(
        "SELECT * FROM ingress_payload WHERE event_id = ?", (event_id,)
    )
    row = cur.fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


class TestPureHelpers:
    def test_canonicalize_update_is_deterministic_sorted_json(self):
        upd = _update()
        a = canonicalize_update(upd)
        b = canonicalize_update(upd)
        assert a == b

    def test_canonicalize_update_is_compact_no_whitespace_padding(self):
        upd = _update()
        a = canonicalize_update(upd)
        assert a == a.strip()  # no incidental whitespace padding
        assert b'": ' not in a  # compact separators, not the default json.dumps spacing

    def test_canonicalize_update_rejects_nan_and_infinity(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            fake_update = SimpleNamespace(to_dict=lambda bad=bad: {"x": bad})
            with pytest.raises(ValueError):
                canonicalize_update(fake_update)

    def test_event_id_format(self):
        assert compute_event_id("default", 777, 1000) == "telegram:default:777:1000"

    def test_payload_hash_format(self):
        h = compute_payload_hash(b"abc")
        assert h.startswith("sha256:")
        assert len(h) == len("sha256:") + 64

    def test_general_topic_normalizes_to_null(self):
        assert normalize_thread_id("1") is None
        assert normalize_thread_id(None) is None
        assert normalize_thread_id("271") == 271

    def test_classify_event_type_command(self):
        assert classify_event_type(_message(text="/start")) == "command"

    def test_classify_event_type_text(self):
        assert classify_event_type(_message(text="hello")) == "text"

    def test_classify_event_type_other_when_no_recognized_content(self):
        assert classify_event_type(_message(text=None)) == "other"

    def test_capture_authorization_excludes_pairing_passthrough_dm(self):
        from gateway.config import Platform, PlatformConfig
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = object.__new__(TelegramAdapter)
        adapter.platform = Platform.TELEGRAM
        adapter.config = PlatformConfig(
            enabled=True,
            token="fake-token",
            extra={"allow_from": ["111"], "unauthorized_dm_behavior": "pair"},
        )
        adapter._bot = SimpleNamespace(id=999, username="capture_bot")
        adapter._message_handler = None
        message = _message(
            chat=_chat(chat_id=333, is_forum=False, chat_type="private"),
            from_user=_user(user_id=333),
            thread_id=None,
        )

        assert adapter._is_user_authorized_from_message(message) is True
        assert adapter._is_capture_sender_authorized(message) is False

    def test_capture_authorization_uses_registered_profile_bound_callback(self):
        from gateway.config import Platform, PlatformConfig
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = object.__new__(TelegramAdapter)
        adapter.platform = Platform.TELEGRAM
        adapter.config = PlatformConfig(enabled=True, token="fake-token", extra={})
        adapter._bot = SimpleNamespace(id=999, username="capture_bot")
        adapter._message_handler = lambda _event: None
        adapter._authorization_check = lambda user_id, chat_type, chat_id: False
        message = _message(
            chat=_chat(chat_id=333, is_forum=False, chat_type="private"),
            from_user=_user(user_id=333),
            thread_id=None,
        )

        assert adapter._is_capture_sender_authorized(message) is False

    @pytest.mark.parametrize("malformed", [{}, "", 0, None])
    def test_present_falsey_capture_routes_are_configured_and_rejected(self, malformed):
        from gateway.config import Platform, PlatformConfig
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = object.__new__(TelegramAdapter)
        adapter.platform = Platform.TELEGRAM
        adapter.config = PlatformConfig(
            enabled=True, token="fake-token", extra={"capture_routes": malformed}
        )
        adapter._capture_route_table_cache = None

        assert adapter._capture_is_configured() is True
        with pytest.raises(ValueError):
            adapter._capture_route_table()

    def test_capture_store_close_resets_store_and_queue_for_reconnect(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        class _Store:
            def __init__(self):
                self.closed = 0

            def close(self):
                self.closed += 1

        adapter = object.__new__(TelegramAdapter)
        store = _Store()
        adapter._capture_store = store
        adapter._capture_queue = object()

        adapter._close_capture_store()

        assert store.closed == 1
        assert adapter._capture_store is None
        assert adapter._capture_queue is None

        adapter._close_capture_store()
        assert store.closed == 1

    @pytest.mark.asyncio
    async def test_capture_failure_alert_replies_in_capture_topic(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = object.__new__(TelegramAdapter)
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))
        source_message = _message(
            chat=_chat(chat_id=CAPTURE_CHAT_ID),
            thread_id=CAPTURE_THREAD_ID,
            message_id=42,
        )

        await adapter._deliver_capture_failure_alert("disk full", source_message)

        adapter.send.assert_awaited_once_with(
            str(CAPTURE_CHAT_ID),
            "Capture ingress failure: disk full",
            metadata={
                "thread_id": str(CAPTURE_THREAD_ID),
                "telegram_reply_to_message_id": 42,
            },
        )


class TestRoutePolicyTable:
    def test_exact_match_requires_both_chat_and_thread(self):
        table = RoutePolicyTable(_capture_only_routes())
        assert table.lookup(CAPTURE_CHAT_ID, CAPTURE_THREAD_ID) is not None
        # Same thread number, different chat: must not match.
        assert table.lookup(-999, CAPTURE_THREAD_ID) is None
        # Same chat, different thread: must not match.
        assert table.lookup(CAPTURE_CHAT_ID, 999) is None

    def test_general_topic_route_matches_null_thread(self):
        routes = [
            {"chat_id": CAPTURE_CHAT_ID, "thread_id": None, "mode": "capture_only", "sink": "s", "policy_version": "1.0.0"}
        ]
        table = RoutePolicyTable(routes)
        assert table.lookup(CAPTURE_CHAT_ID, None) is not None

    def test_no_configured_route_is_none(self):
        table = RoutePolicyTable([])
        assert table.lookup(CAPTURE_CHAT_ID, CAPTURE_THREAD_ID) is None

    def test_unrecognized_mode_raises_instead_of_becoming_unrouted_passthrough(self):
        routes = [
            {
                "chat_id": CAPTURE_CHAT_ID,
                "thread_id": CAPTURE_THREAD_ID,
                "mode": "Capture_Only",
                "sink": "s",
                "policy_version": "1.0.0",
            }
        ]
        with pytest.raises(ValueError, match="mode"):
            RoutePolicyTable(routes)

    @pytest.mark.parametrize(
        "mutate, expected",
        [
            (lambda route: route.update(extra="x"), "unknown"),
            (lambda route: route.pop("sink"), "required"),
            (lambda route: route.update(chat_id=True), "chat_id"),
            (lambda route: route.update(chat_id=0), "chat_id"),
            (lambda route: route.update(thread_id="271"), "thread_id"),
            (lambda route: route.update(thread_id=0), "thread_id"),
            (lambda route: route.update(sink="Bad Sink"), "sink"),
            (lambda route: route.update(policy_version="v1"), "policy_version"),
        ],
    )
    def test_route_policy_rejects_values_outside_merged_schema(self, mutate, expected):
        route = dict(_capture_only_routes()[0])
        mutate(route)
        with pytest.raises(ValueError, match=expected):
            RoutePolicyTable([route])

    def test_route_policy_rejects_duplicate_normalized_route_key(self):
        first = dict(_capture_only_routes()[0], thread_id=None)
        duplicate = dict(_capture_only_routes()[0], thread_id=1)
        with pytest.raises(ValueError, match="duplicate"):
            RoutePolicyTable([first, duplicate])


class TestCaptureIngressStore:
    def _kwargs(self, **override):
        base = dict(
            event_id="telegram:default:777:1000",
            platform="telegram",
            account_id=ACCOUNT_ID,
            profile=PROFILE,
            update_id=1000,
            chat_id=CAPTURE_CHAT_ID,
            thread_id=CAPTURE_THREAD_ID,
            message_id=42,
            sender_id=555,
            event_type="text",
            received_at="2026-08-06T20:00:00Z",
            payload_hash="sha256:" + "a" * 64,
            route_mode="capture_only",
            sink="braindump",
            payload_json="{}",
        )
        base.update(override)
        return base

    def test_first_insert_returns_inserted_and_persists_row_and_payload(self, store):
        result = store.commit_capture(**self._kwargs())
        assert result == INSERTED
        row = _ledger_row(store, "telegram:default:777:1000")
        assert row["status"] == "pending"
        assert row["attempts"] == 0
        assert row["lease_expires_at"] is None
        assert row["last_error"] is None
        assert row["completed_at"] is None
        payload = _payload_row(store, "telegram:default:777:1000")
        assert payload["payload_json"] == "{}"
        assert payload["payload_format"] == "telegram-update-json-v1"

    def test_identical_duplicate_is_a_noop_first_fields_preserved(self, store):
        store.commit_capture(**self._kwargs(received_at="2026-08-06T20:00:00Z"))
        result = store.commit_capture(**self._kwargs(received_at="2026-08-06T20:05:00Z"))
        assert result == DUPLICATE_SAME
        row = _ledger_row(store, "telegram:default:777:1000")
        assert row["received_at"] == "2026-08-06T20:00:00Z"  # first write wins, not overwritten

    def test_duplicate_same_rejects_missing_companion_payload(self, store):
        kwargs = self._kwargs()
        store.commit_capture(**kwargs)
        store._conn.execute("DELETE FROM ingress_payload WHERE event_id = ?", (kwargs["event_id"],))

        with pytest.raises(CapturePersistenceError):
            store.commit_capture(**kwargs)

    def test_rollback_failure_does_not_mask_normalized_persistence_error(self, store):
        real = store._conn

        class _RollbackFailingConnection:
            @property
            def in_transaction(self):
                return real.in_transaction

            def execute(self, sql, parameters=()):
                if sql == "ROLLBACK":
                    raise sqlite3.OperationalError("rollback failed")
                if sql.startswith("INSERT INTO ingress_payload"):
                    raise sqlite3.OperationalError("payload insert failed")
                return real.execute(sql, parameters)

        store._conn = _RollbackFailingConnection()
        try:
            with pytest.raises(CapturePersistenceError, match="payload insert failed"):
                store.commit_capture(**self._kwargs())
        finally:
            store._conn = real

    def test_conflicting_duplicate_payload_raises_and_does_not_overwrite(self, store):
        store.commit_capture(**self._kwargs(payload_hash="sha256:" + "a" * 64))
        with pytest.raises(RouteConflict):
            store.commit_capture(**self._kwargs(payload_hash="sha256:" + "b" * 64))
        row = _ledger_row(store, "telegram:default:777:1000")
        assert row["payload_hash"] == "sha256:" + "a" * 64  # unchanged

    def test_integrity_error_unrelated_to_a_duplicate_race_is_persistence_error_not_conflict(self, store):
        """A NOT-NULL violation (e.g. a caller passing account_id=None,
        which can happen if commit_capture is ever reached before the bot's
        own id is known) is a genuine storage/programming-adjacent failure,
        not a lost race against a concurrent insert of the same event_id --
        raising RouteConflict here would misdiagnose it and mislead
        debugging/alerting, even though both exception types fail closed the
        same way (no delegation, no ack).
        """
        with pytest.raises(CapturePersistenceError):
            store.commit_capture(**self._kwargs(account_id=None))
        assert _ledger_row(store, "telegram:default:777:1000") is None

    def test_injected_storage_failure_raises_capture_persistence_error(self, store):
        store._conn.close()  # simulate a hard storage failure (closed handle)
        with pytest.raises(CapturePersistenceError):
            store.commit_capture(**self._kwargs())

    def test_begin_immediate_failure_surfaces_persistence_error_not_a_secondary_rollback_error(
        self, store, db_path
    ):
        """A real lock-contention failure (not mocked): a second connection
        holds SQLite's one write lock (even under WAL, writers still
        serialize) so the store's own BEGIN IMMEDIATE fails before any
        transaction of its own ever opens. Rolling back a transaction that
        was never started raises its own sqlite3 error ("cannot rollback -
        no transaction is active"), which must not mask the original
        failure or escape uncaught as a different, uncategorized error --
        CapturePersistenceError must be exactly what surfaces.
        """
        blocker = sqlite3.connect(str(db_path), isolation_level=None, timeout=0)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(CapturePersistenceError):
                store.commit_capture(**self._kwargs())
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
        assert _ledger_row(store, "telegram:default:777:1000") is None

    def test_foreign_key_enforcement_rejects_orphan_payload_row(self, store):
        """Defense-in-depth: commit_capture always inserts ledger-then-payload
        atomically, so this never fires in normal operation -- but the FK
        constraint declared in the schema (ingress_payload.event_id
        REFERENCES ingress_ledger.event_id) must actually be enforced, not
        merely decorative (SQLite requires PRAGMA foreign_keys=ON per
        connection; without it, the REFERENCES clause is silently a no-op).
        """
        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(
                "INSERT INTO ingress_payload (event_id, payload_hash, payload_format, payload_json) "
                "VALUES (?, ?, ?, ?)",
                ("telegram:default:777:9999", "sha256:" + "a" * 64, "telegram-update-json-v1", "{}"),
            )


class TestCaptureAwareQueueDispatchGating:
    @pytest.mark.asyncio
    async def test_capture_only_route_commits_then_terminal_deny(self, store):
        queue = _make_queue(store, _capture_only_routes())
        upd = _update(update_id=1, text="hello")

        await queue.put(upd)

        assert queue.qsize() == 0  # never delegated to the underlying queue
        eid = compute_event_id(PROFILE, ACCOUNT_ID, 1)
        row = _ledger_row(store, eid)
        assert row is not None
        assert row["route_mode"] == "capture_only"

    @pytest.mark.asyncio
    async def test_sender_identity_less_message_on_capture_only_route_still_denied(self, store):
        """A message-like update with no from_user (e.g. a channel post,
        whose identity lives in sender_chat instead) cannot be keyed into a
        ledger row -- the ingress-ledger contract requires a non-null human
        sender_id -- but a capture-only route's terminal deny is an
        unconditional owner directive ("Capture must never start an agent
        turn... for the capture-only topic"), not conditioned on whether a
        ledger row could be produced. It must still be denied, not silently
        delegated to dispatch just because it couldn't be captured.
        """
        queue = _make_queue(store, _capture_only_routes())
        msg = _message(text="channel announcement", from_user=None)
        msg = Message(
            message_id=msg.message_id,
            date=msg.date,
            chat=msg.chat,
            from_user=None,
            sender_chat=_chat(chat_id=-500, chat_type="channel"),
            text=msg.text,
            message_thread_id=msg.message_thread_id,
            is_topic_message=msg.is_topic_message,
        )
        upd = _update(update_id=16, message=msg)

        await queue.put(upd)

        eid = compute_event_id(PROFILE, ACCOUNT_ID, 16)
        assert _ledger_row(store, eid) is None  # cannot be captured (no human sender_id)
        assert queue.qsize() == 0  # but must NOT be delegated to dispatch either

    @pytest.mark.asyncio
    async def test_sender_identity_less_message_on_agent_route_passes_through(self, store):
        """Same identity-less shape, but on an 'agent' route: existing
        (non-capture) dispatch behavior is unaffected -- this envelope only
        adds a new hard deny for capture_only, never a new block for agent.
        """
        queue = _make_queue(store, _agent_routes())
        msg = _message(text="channel announcement", from_user=None)
        msg = Message(
            message_id=msg.message_id,
            date=msg.date,
            chat=msg.chat,
            from_user=None,
            sender_chat=_chat(chat_id=-500, chat_type="channel"),
            text=msg.text,
            message_thread_id=msg.message_thread_id,
            is_topic_message=msg.is_topic_message,
        )
        upd = _update(update_id=17, message=msg)

        await queue.put(upd)

        eid = compute_event_id(PROFILE, ACCOUNT_ID, 17)
        assert _ledger_row(store, eid) is None
        assert queue.qsize() == 1  # unchanged existing behavior

    @pytest.mark.asyncio
    async def test_command_on_capture_only_route_captured_as_inert_text_not_dispatched(self, store):
        queue = _make_queue(store, _capture_only_routes())
        upd = _update(update_id=2, text="/deploy prod")

        await queue.put(upd)

        assert queue.qsize() == 0
        eid = compute_event_id(PROFILE, ACCOUNT_ID, 2)
        row = _ledger_row(store, eid)
        assert row["event_type"] == "command"

    @pytest.mark.asyncio
    async def test_media_on_capture_only_route(self, store):
        from telegram import PhotoSize

        queue = _make_queue(store, _capture_only_routes())
        msg = _message(text=None, photo=[PhotoSize(file_id="f1", file_unique_id="u1", width=10, height=10)])
        upd = _update(update_id=3, message=msg)

        await queue.put(upd)

        eid = compute_event_id(PROFILE, ACCOUNT_ID, 3)
        row = _ledger_row(store, eid)
        assert row["event_type"] == "media"
        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_location_on_capture_only_route(self, store):
        from telegram import Location

        queue = _make_queue(store, _capture_only_routes())
        msg = _message(text=None, location=Location(longitude=1.0, latitude=2.0))
        upd = _update(update_id=4, message=msg)

        await queue.put(upd)

        eid = compute_event_id(PROFILE, ACCOUNT_ID, 4)
        row = _ledger_row(store, eid)
        assert row["event_type"] == "location"
        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_other_message_like_content_kind(self, store):
        queue = _make_queue(store, _capture_only_routes())
        # message_id/from_user present, no text/media/location: e.g. a
        # message the ledger contract still requires a row for.
        msg = _message(text=None)
        upd = _update(update_id=5, message=msg)

        await queue.put(upd)

        eid = compute_event_id(PROFILE, ACCOUNT_ID, 5)
        row = _ledger_row(store, eid)
        assert row["event_type"] == "other"

    @pytest.mark.asyncio
    async def test_non_message_like_update_passes_through_unchanged(self, store):
        queue = _make_queue(store, _capture_only_routes())
        cq = CallbackQuery(id="cbq1", from_user=_user(), chat_instance="ci1", data="x")
        upd = Update(update_id=6, callback_query=cq)

        await queue.put(upd)

        assert queue.qsize() == 1  # delegated unchanged, not captured
        assert queue.get_nowait() is upd

    @pytest.mark.asyncio
    async def test_agent_route_commits_then_delegates_exactly_once(self, store):
        queue = _make_queue(store, _agent_routes())
        upd = _update(update_id=7)

        await queue.put(upd)

        eid = compute_event_id(PROFILE, ACCOUNT_ID, 7)
        row = _ledger_row(store, eid)
        assert row["route_mode"] == "agent"
        assert queue.qsize() == 1
        assert queue.get_nowait() is upd

    @pytest.mark.asyncio
    async def test_identical_agent_retry_does_not_delegate_twice(self, store):
        queue = _make_queue(store, _agent_routes())
        upd = _update(update_id=70)

        await queue.put(upd)
        await queue.put(upd)

        assert queue.qsize() == 1
        assert queue.get_nowait() is upd

    @pytest.mark.asyncio
    async def test_drop_route_no_ledger_row_passthrough(self, store):
        queue = _make_queue(store, _drop_routes())
        upd = _update(update_id=8)

        await queue.put(upd)

        eid = compute_event_id(PROFILE, ACCOUNT_ID, 8)
        assert _ledger_row(store, eid) is None
        assert queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_no_configured_route_passthrough(self, store):
        queue = _make_queue(store, [])  # nothing configured for this chat/topic
        upd = _update(update_id=9)

        await queue.put(upd)

        eid = compute_event_id(PROFILE, ACCOUNT_ID, 9)
        assert _ledger_row(store, eid) is None
        assert queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_exact_route_matching_same_thread_number_other_chat_no_match(self, store):
        queue = _make_queue(store, _capture_only_routes())
        msg = _message(chat=_chat(chat_id=-2002), thread_id=CAPTURE_THREAD_ID)
        upd = _update(update_id=10, message=msg)

        await queue.put(upd)

        eid = compute_event_id(PROFILE, ACCOUNT_ID, 10)
        assert _ledger_row(store, eid) is None
        assert queue.qsize() == 1  # unmatched: passes through like an unrouted chat

    @pytest.mark.asyncio
    async def test_identical_duplicate_no_second_row_capture_only_still_denied(self, store):
        queue = _make_queue(store, _capture_only_routes())
        upd = _update(update_id=11, text="hello")

        await queue.put(upd)
        await queue.put(upd)  # simulated redelivery of the same update

        eid = compute_event_id(PROFILE, ACCOUNT_ID, 11)
        cur = store._conn.execute(
            "SELECT COUNT(*) FROM ingress_ledger WHERE event_id = ?", (eid,)
        )
        assert cur.fetchone()[0] == 1
        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_conflicting_duplicate_fails_closed_no_delegation(self, store):
        queue = _make_queue(store, _agent_routes())
        msg_a = _message(message_id=42, text="hello")
        upd_a = Update(update_id=12, message=msg_a)
        await queue.put(upd_a)
        assert queue.qsize() == 1
        queue.get_nowait()

        # Same update_id, different canonical payload (different text).
        msg_b = _message(message_id=42, text="different content")
        upd_b = Update(update_id=12, message=msg_b)
        with pytest.raises(RouteConflict):
            await queue.put(upd_b)
        assert queue.qsize() == 0  # no delegation on conflict

    @pytest.mark.asyncio
    async def test_injected_storage_failure_no_delegation(self, store):
        alerts = []
        queue = _make_queue(store, _capture_only_routes(), alert=alerts.append)
        store.close()  # force every commit to fail
        upd = _update(update_id=13)

        with pytest.raises(CapturePersistenceError):
            await queue.put(upd)

        assert queue.qsize() == 0
        assert alerts  # deterministic failure was alerted

    @pytest.mark.asyncio
    async def test_unauthorized_sender_on_capture_only_route_is_neither_captured_nor_delegated(self, store):
        queue = _make_queue(store, _capture_only_routes(), is_authorized=False)
        upd = _update(update_id=14)

        await queue.put(upd)

        eid = compute_event_id(PROFILE, ACCOUNT_ID, 14)
        assert _ledger_row(store, eid) is None
        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_unauthorized_sender_on_agent_route_preserves_downstream_pairing_behavior(self, store):
        queue = _make_queue(store, _agent_routes(), is_authorized=False)
        upd = _update(update_id=140)

        await queue.put(upd)

        eid = compute_event_id(PROFILE, ACCOUNT_ID, 140)
        assert _ledger_row(store, eid) is None
        assert queue.qsize() == 1
        assert queue.get_nowait() is upd

    @pytest.mark.asyncio
    async def test_any_bot_authored_message_on_capture_only_route_is_denied(self, store):
        queue = _make_queue(store, _capture_only_routes(), is_own=False, is_authorized=True)
        upd = _update(update_id=15, from_user=_user(user_id=888, is_bot=True))

        await queue.put(upd)

        eid = compute_event_id(PROFILE, ACCOUNT_ID, 15)
        assert _ledger_row(store, eid) is None
        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_queue_sentinel_passes_through_unchanged(self, store):
        queue = _make_queue(store, _capture_only_routes())
        sentinel = object()

        await queue.put(sentinel)

        assert queue.qsize() == 1
        assert queue.get_nowait() is sentinel

    @pytest.mark.asyncio
    async def test_polling_batch_partial_failure_then_retry_collapses_first_inserts_second_once(self, store):
        queue = _make_queue(store, _capture_only_routes())
        upd_n = _update(update_id=20, message_id=100, text="first")
        upd_n1 = _update(update_id=21, message_id=101, text="second")

        await queue.put(upd_n)  # N commits

        store.close()  # simulate N+1 failing to persist
        with pytest.raises(CapturePersistenceError):
            await queue.put(upd_n1)

        # Reopen (as a fresh retry attempt would) and replay both.
        reopened = CaptureIngressStore(store._db_path)
        try:
            retry_queue = _make_queue(reopened, _capture_only_routes())
            await retry_queue.put(upd_n)  # redelivered N: collapses, no second row
            await retry_queue.put(upd_n1)  # N+1 inserts once now that storage recovered

            eid_n = compute_event_id(PROFILE, ACCOUNT_ID, 20)
            eid_n1 = compute_event_id(PROFILE, ACCOUNT_ID, 21)
            assert reopened._conn.execute(
                "SELECT COUNT(*) FROM ingress_ledger WHERE event_id = ?", (eid_n,)
            ).fetchone()[0] == 1
            assert reopened._conn.execute(
                "SELECT COUNT(*) FROM ingress_ledger WHERE event_id = ?", (eid_n1,)
            ).fetchone()[0] == 1
        finally:
            reopened.close()


class TestPollingRetryCompatibility:
    """Production-shaped: drives PTB's REAL network_retry_loop (not a
    reimplementation) with a polling_action_cb shaped exactly like PTB
    22.6's actual ``Updater.start_polling`` closure -- get a batch, loop
    ``await update_queue.put(update)`` over it, only then advance the
    offset. Verified directly against PTB 22.6 source
    (telegram.ext._utils.networkloop.network_retry_loop): its outer retry
    loop catches only RetryAfter/TimedOut/InvalidToken/TelegramError. Any
    other exception type from ``put()`` propagates straight out on the
    first attempt, killing the polling task instead of entering the
    bounded retry ladder -- exactly what this test would catch.
    """

    @staticmethod
    async def _drive_real_ptb_retry_loop(queue, updates, *, max_retries):
        from telegram.ext._utils.networkloop import network_retry_loop

        state = {"last_update_id": 0, "attempts": 0}
        errors_seen = []

        async def polling_action_cb():
            state["attempts"] += 1
            for u in updates:
                await queue.put(u)  # the seam under test
            state["last_update_id"] = updates[-1].update_id + 1

        exc_raised = None
        try:
            await network_retry_loop(
                action_cb=polling_action_cb,
                on_err_cb=errors_seen.append,
                description="test polling",
                interval=0,
                max_retries=max_retries,
                repeat_on_success=False,
            )
        except Exception as exc:  # the loop re-raises once max_retries is exhausted
            exc_raised = exc
        return state, errors_seen, exc_raised

    @pytest.mark.asyncio
    async def test_persistence_failure_enters_bounded_retry_not_immediate_crash(self, store):
        from telegram.error import TelegramError

        queue = _make_queue(store, _capture_only_routes())
        store.close()  # every commit_capture call will now fail
        upd = _update(update_id=99, text="hello")

        state, errors_seen, exc_raised = await self._drive_real_ptb_retry_loop(
            queue, [upd], max_retries=2,
        )

        # Bounded: the real PTB loop retried (didn't die on attempt 1), then
        # gave up and re-raised once max_retries was exhausted.
        assert state["attempts"] == 3  # initial attempt + 2 retries
        assert state["last_update_id"] == 0  # offset never advanced
        assert len(errors_seen) == 3
        assert all(isinstance(e, TelegramError) for e in errors_seen)
        assert isinstance(exc_raised, TelegramError)
        assert isinstance(exc_raised, CapturePersistenceError)

    @pytest.mark.asyncio
    async def test_conflicting_duplicate_enters_bounded_retry_not_immediate_crash(self, store):
        from telegram.error import TelegramError

        queue = _make_queue(store, _agent_routes())
        first = _update(update_id=100, message_id=42, text="hello")
        await queue.put(first)  # commits and delegates once
        queue.get_nowait()

        conflicting = _update(update_id=100, message_id=42, text="different content")

        state, errors_seen, exc_raised = await self._drive_real_ptb_retry_loop(
            queue, [conflicting], max_retries=1,
        )

        assert state["attempts"] == 2
        assert state["last_update_id"] == 0
        assert all(isinstance(e, TelegramError) for e in errors_seen)
        assert isinstance(exc_raised, TelegramError)
        assert isinstance(exc_raised, RouteConflict)

    @pytest.mark.asyncio
    async def test_persistence_failure_does_not_delegate_to_underlying_queue_across_retries(self, store):
        queue = _make_queue(store, _capture_only_routes())
        store.close()
        upd = _update(update_id=101, text="hello")

        await self._drive_real_ptb_retry_loop(queue, [upd], max_retries=1)

        assert queue.qsize() == 0  # never delegated on any attempt

    def test_capture_persistence_error_and_route_conflict_are_distinct_telegram_errors(self):
        from telegram.error import TelegramError

        assert issubclass(CapturePersistenceError, TelegramError)
        assert issubclass(RouteConflict, TelegramError)
        assert not issubclass(CapturePersistenceError, RouteConflict)
        assert not issubclass(RouteConflict, CapturePersistenceError)
        # Constructible the same way PTB's own TelegramError subclasses are
        # (single message string), so PTB's logging (str(exc)) still works.
        assert str(CapturePersistenceError("boom")) == "boom"
        assert str(RouteConflict("boom")) == "boom"


class TestConnectFailsClosedBeforeTouchingPTB:
    """Slice 1.1R-B: route-policy validation and receiving_profile identity
    must be rejected *before* dispatch/admission -- i.e. at connect() time,
    before the adapter ever touches PTB's ApplicationBuilder -- not
    discovered lazily on the first message. Scoped to
    ``_capture_is_configured()``: an adapter with no ``capture_routes`` at
    all must connect exactly as it did before this slice.
    """

    @staticmethod
    def _adapter(extra):
        from gateway.config import PlatformConfig
        from plugins.platforms.telegram.adapter import TelegramAdapter

        return TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra=extra))

    @pytest.mark.asyncio
    async def test_invalid_capture_routes_fails_before_touching_ptb(self, monkeypatch):
        import plugins.platforms.telegram.adapter as tg_adapter

        builder_calls = []
        monkeypatch.setattr(
            tg_adapter,
            "Application",
            SimpleNamespace(builder=lambda: builder_calls.append(1)),
        )
        adapter = self._adapter(
            {"capture_routes": [{"chat_id": -1, "thread_id": None, "mode": "bogus", "sink": "s", "policy_version": "1.0.0"}]}
        )

        ok = await adapter.connect()

        assert ok is False
        assert builder_calls == []  # never reached PTB at all
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_code == "invalid_capture_routes"
        assert adapter.fatal_error_retryable is False

    @pytest.mark.asyncio
    async def test_configured_but_unstamped_profile_fails_before_touching_ptb(self, monkeypatch):
        import plugins.platforms.telegram.adapter as tg_adapter

        builder_calls = []
        monkeypatch.setattr(
            tg_adapter,
            "Application",
            SimpleNamespace(builder=lambda: builder_calls.append(1)),
        )
        adapter = self._adapter({"capture_routes": _capture_only_routes()})
        assert adapter.receiving_profile is None  # never stamped

        ok = await adapter.connect()

        assert ok is False
        assert builder_calls == []
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_code == "missing_receiving_profile"
        assert adapter.fatal_error_retryable is False

    @pytest.mark.asyncio
    async def test_valid_capture_routes_and_stamped_profile_reaches_ptb(self, monkeypatch):
        """Positive control: both guards pass and connect() proceeds into
        the real PTB builder chain (proven by reaching Application.builder()
        -- connect()'s own outer try/except catches whatever the stubbed
        builder raises and returns False, same as any real connect
        failure, so this only asserts the guards let it through)."""
        import plugins.platforms.telegram.adapter as tg_adapter

        builder_calls = []

        def _builder():
            builder_calls.append(1)
            raise RuntimeError("stop here -- past the capture guards")

        monkeypatch.setattr(tg_adapter, "Application", SimpleNamespace(builder=_builder))
        adapter = self._adapter({"capture_routes": _capture_only_routes()})
        adapter.receiving_profile = "coder"

        ok = await adapter.connect()

        assert ok is False  # the stubbed builder failure, not a capture guard
        assert builder_calls == [1]
        assert adapter.fatal_error_code != "invalid_capture_routes"
        assert adapter.fatal_error_code != "missing_receiving_profile"

    @pytest.mark.asyncio
    async def test_no_capture_routes_configured_never_requires_profile(self, monkeypatch):
        """Regression guard: an adapter with no capture_routes at all must
        not be blocked by either check -- same as before this slice."""
        import plugins.platforms.telegram.adapter as tg_adapter

        builder_calls = []

        def _builder():
            builder_calls.append(1)
            raise RuntimeError("stop here -- past the capture guards")

        monkeypatch.setattr(tg_adapter, "Application", SimpleNamespace(builder=_builder))
        adapter = self._adapter({})
        assert adapter.receiving_profile is None

        ok = await adapter.connect()

        assert ok is False
        assert builder_calls == [1]  # reached PTB despite no stamped profile
        assert adapter.fatal_error_code != "missing_receiving_profile"

    def test_capture_route_table_is_cached_not_rebuilt_per_call(self):
        adapter = self._adapter({"capture_routes": _capture_only_routes()})
        first = adapter._capture_route_table()
        second = adapter._capture_route_table()
        assert first is second

    def test_capture_route_table_validation_error_surfaces_on_first_access(self):
        adapter = self._adapter(
            {"capture_routes": [{"chat_id": -1, "thread_id": None, "mode": "bogus", "sink": "s", "policy_version": "1.0.0"}]}
        )
        with pytest.raises(ValueError, match="mode"):
            adapter._capture_route_table()


class TestDisconnectClosesCaptureStore:
    """Slice 1.1R-B blocker 4: disconnect() must close the capture ledger
    (after PTB update admission has already stopped) and clear both the
    store and queue references, so a later connect() builds fresh ones
    instead of reusing a closed sqlite3 connection.
    """

    @pytest.mark.asyncio
    async def test_disconnect_closes_store_and_clears_references(self, tmp_path, monkeypatch):
        from gateway.config import PlatformConfig
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = TelegramAdapter(
            PlatformConfig(enabled=True, token="test-token", extra={"capture_db_path": str(tmp_path / "c.db")})
        )
        real_store = adapter._ensure_capture_store()
        adapter._capture_queue = adapter._build_capture_queue()
        assert real_store._closed is False

        await adapter.disconnect()

        assert real_store._closed is True
        assert adapter._capture_store is None
        assert adapter._capture_queue is None

    @pytest.mark.asyncio
    async def test_disconnect_is_safe_when_capture_store_was_never_created(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
        assert adapter._capture_store is None

        await adapter.disconnect()  # must not raise

        assert adapter._capture_store is None
        assert adapter._capture_queue is None

    @pytest.mark.asyncio
    async def test_reconnect_after_disconnect_builds_a_fresh_store_not_a_closed_one(self, tmp_path):
        from gateway.config import PlatformConfig
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = TelegramAdapter(
            PlatformConfig(enabled=True, token="test-token", extra={"capture_db_path": str(tmp_path / "c.db")})
        )
        first_store = adapter._ensure_capture_store()
        await adapter.disconnect()
        assert first_store._closed is True

        second_store = adapter._ensure_capture_store()

        assert second_store is not first_store
        assert second_store._closed is False
        # Proves it's a live, usable connection, not the closed one.
        second_store.commit_capture(
            event_id="telegram:default:1:1", platform="telegram", account_id=1,
            profile="default", update_id=1, chat_id=-1, thread_id=None,
            message_id=1, sender_id=1, event_type="text", received_at="2026-01-01T00:00:00Z",
            payload_hash="sha256:" + "a" * 64, route_mode="capture_only", sink="s",
            payload_json="{}",
        )
        second_store.close()

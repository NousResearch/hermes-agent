"""A shutdown flush must record a key the next boot's recovery can resolve (#136189).

``_pending_messages`` slots are keyed by the adapter's own isolation flags
(``PlatformConfig.extra``), while ``SessionStore`` routes by the gateway-wide
``group_sessions_per_user`` / ``thread_sessions_per_user``. A platform-block override reaches
``PlatformConfig.extra`` (``PlatformConfig.from_dict`` promotes untyped platform keys) and the
runner only ``setdefault``s the gateway-wide value, so one platform can legitimately key a
conversation differently from the store — and a standalone gateway hands that same key to the
busy handler, so the queue head and the FIFO tail inherit it. ``recover_pending_to_db`` resolves
a payload's key through the store alone — routing map first, then the durable row under the exact
key — so a flush written under the adapter key matched neither: the message stranded in the spool
with a "Cannot recover pending message" warning on every boot.

These tests drive the real shutdown entry points (``adapter.cancel_background_tasks()``,
``runner.stop()``) and the real recovery pass, with the two derivations disagreeing.
"""

import asyncio
import json
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.yuanbao import RecallGuardMiddleware
from gateway.session import SessionStore
from gateway.shutdown_flush import flush_pending_to_file, recover_pending_to_db
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source

# One platform overrides per-user thread isolation; the store keeps the gateway-wide default
# (threads shared unless ``thread_sessions_per_user``). Same conversation, two keys.
ADAPTER_KEY = "agent:main:telegram:group:900:900:u1"
STORE_KEY = "agent:main:telegram:group:900:900"


class _FakeGatewayDB:
    """Minimal SessionDB stand-in: the exact-key peer finder the resolver uses, plus row appends."""

    def __init__(self, rows_by_key=None):
        self.rows_by_key = rows_by_key or {}
        self.appended = []

    def find_latest_gateway_session_for_peer(self, *, source, session_key=None, **kwargs):
        return self.rows_by_key.get(session_key)

    def append_message(self, **kwargs):
        self.appended.append(kwargs)


def _store(tmp_path, db, *, config=None):
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path / "sessions", config=config or GatewayConfig())
    store._loaded = True
    store._db = db  # pin: _db_for_key returns this db for every key (no profile I/O)
    return store


@pytest.fixture
def flush_dir(tmp_path, monkeypatch):
    directory = tmp_path / "pending_messages"
    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("gateway.shutdown_flush._get_flush_dir", lambda: directory)
    return directory


def _queued_event() -> MessageEvent:
    return MessageEvent(
        text="queued while busy",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="900", chat_type="group", thread_id="900"),
        message_id="queued-1",
    )


def _payload_keys(flush_dir) -> list[str]:
    return sorted(json.loads(p.read_text())["session_key"]
                  for p in flush_dir.glob("pending-*.json"))


def _spooled_files(flush_dir) -> list:
    return list(flush_dir.glob("pending-*.json"))


def _wire(tmp_path, flush_dir, *, store_db=None):
    """A restart-test runner + adapter whose platform flags disagree with the store's."""
    runner, adapter = make_restart_runner()
    store = _store(tmp_path, store_db if store_db is not None else _FakeGatewayDB())
    runner.session_store = store
    adapter.set_session_store(store)
    adapter.config.extra["thread_sessions_per_user"] = True
    return runner, adapter, store


def test_the_two_derivations_disagree_for_the_same_source(tmp_path, flush_dir):
    """Precondition of the whole bug: the slot key and the store key are different strings."""
    _runner, adapter, store = _wire(tmp_path, flush_dir)
    event = _queued_event()
    assert adapter._event_session_key(event) == ADAPTER_KEY
    assert store._generate_session_key(event.source) == STORE_KEY


@pytest.mark.asyncio
async def test_adapter_shutdown_flush_records_the_store_key(tmp_path, flush_dir):
    _runner, adapter, _ = _wire(tmp_path, flush_dir)
    event = _queued_event()
    adapter._pending_messages[ADAPTER_KEY] = event

    await adapter.cancel_background_tasks()

    assert _payload_keys(flush_dir) == [STORE_KEY]


@pytest.mark.asyncio
async def test_adapter_keyed_flush_is_recovered_on_the_next_boot(tmp_path, flush_dir):
    """End to end: the spool a restart reads back lands in the session the store routes to."""
    db = _FakeGatewayDB({STORE_KEY: {"id": "sid-900", "started_at": 0.0}})
    _runner, adapter, store = _wire(tmp_path, flush_dir, store_db=db)
    adapter._pending_messages[ADAPTER_KEY] = _queued_event()

    await adapter.cancel_background_tasks()
    recovered = recover_pending_to_db(db, session_resolver=store.resolve_session_id_for_key)

    assert recovered == 1
    assert [call["session_id"] for call in db.appended] == ["sid-900"]
    assert [call["content"] for call in db.appended] == ["queued while busy"]
    assert _spooled_files(flush_dir) == []


@pytest.mark.asyncio
async def test_runner_shutdown_flush_records_the_store_key(tmp_path, flush_dir):
    """The runner's own stop() flushes the queue head and the FIFO tail the same way."""
    runner, adapter, _ = _wire(tmp_path, flush_dir)
    event = _queued_event()
    adapter_key = adapter._event_session_key(event)
    runner._pending_messages[adapter_key] = event
    runner._queued_events = {adapter_key: [event]}

    with patch("gateway.status.remove_pid_file"), patch(
        "gateway.status.publish_runtime_status"
    ):
        await runner.stop()

    assert _payload_keys(flush_dir) == [STORE_KEY, STORE_KEY]


def test_a_slot_without_a_source_keeps_its_key(tmp_path, flush_dir):
    """Raw-str slots (and agent-history payloads) have no source to re-derive from."""
    def _unusable(_source):
        raise RuntimeError("no session store wired")

    assert flush_pending_to_file({"agent:main:telegram:dm:7": "raw text"},
                                 session_key_for=_unusable) == 1
    assert _payload_keys(flush_dir) == ["agent:main:telegram:dm:7"]


def test_a_failing_derivation_keeps_the_slot_key(tmp_path, flush_dir):
    """A derivation that raises must never cost the payload its place in the spool."""
    def _unusable(_source):
        raise RuntimeError("no session store wired")

    assert flush_pending_to_file({ADAPTER_KEY: _queued_event()},
                                 session_key_for=_unusable) == 1
    assert _payload_keys(flush_dir) == [ADAPTER_KEY]


@pytest.mark.parametrize("derived", ["", 0, None, ["agent:main:telegram:group:900:900"]])
def test_a_derivation_result_that_names_no_session_keeps_the_slot_key(tmp_path, flush_dir,
                                                                     derived):
    """Only a non-empty str is a key the store can route by; anything else keeps the slot key.

    A derivation that returns a list, a number, ``None``, or an empty string would otherwise
    record a payload the next boot's resolution cannot look up — the strand this PR fixes.
    """
    assert flush_pending_to_file({ADAPTER_KEY: _queued_event()},
                                 session_key_for=lambda _source: derived) == 1
    assert _payload_keys(flush_dir) == [ADAPTER_KEY]


# --- A supported synthetic producer whose value keeps the slot key's owner --------------------
#
# ``RecallGuardMiddleware._interrupt_for_recall`` (yuanbao.py's branch C) writes its notice
# straight into the matched in-flight slot, and a recall callback may arrive with no
# ``from_account`` (``DecodeMiddleware.parse_json_push`` admits it). The notice's ``source`` is
# then partial, so its own derivation drops the participant the slot key holds and names a key no
# session was ever created under — re-deriving there strands the notice. The slot itself was
# filled by the ordinary dispatch, so its key is the one the store routes and must be kept.
RECALL_SLOT_KEY = "agent:main:telegram:group:group:G:main:u1"
RECALL_PARTIAL_KEY = "agent:main:telegram:group:group:G:main"


def _wire_aligned(tmp_path, db):
    """Adapter and store agree on per-user isolation: the matched slot key is a routed key."""
    runner, adapter = make_restart_runner()
    store = _store(tmp_path, db, config=GatewayConfig(thread_sessions_per_user=True))
    runner.session_store = store
    adapter.set_session_store(store)
    adapter.config.extra["group_sessions_per_user"] = True
    adapter.config.extra["thread_sessions_per_user"] = True
    return runner, adapter, store


def _queue_recall_notice(adapter, recalled_id: str = "recalled-1") -> MessageEvent:
    """Drive the real recall producer against the maps the ordinary dispatch would have filled."""
    matched = make_restart_source(chat_id="group:G", chat_type="group", thread_id="main")
    assert adapter._event_session_key(MessageEvent(
        text="", message_type=MessageType.TEXT, source=matched)) == RECALL_SLOT_KEY
    adapter._active_sessions[RECALL_SLOT_KEY] = asyncio.Event()
    adapter._processing_msg_ids = {RECALL_SLOT_KEY: recalled_id}
    adapter._processing_msg_texts = {}  # blank: no redaction polling for a notice under test
    RecallGuardMiddleware._interrupt_for_recall(adapter, RECALL_SLOT_KEY, recalled_id, "G", "")
    return adapter._pending_messages[RECALL_SLOT_KEY]


@pytest.mark.asyncio
async def test_a_recall_notice_with_a_partial_source_keeps_its_slot_key(tmp_path, flush_dir):
    """The notice keeps the key it was queued under, and recovery lands it in that session."""
    db = _FakeGatewayDB({RECALL_SLOT_KEY: {"id": "sid-recall", "started_at": 0.0}})
    _runner, adapter, store = _wire_aligned(tmp_path, db)
    notice = _queue_recall_notice(adapter)
    assert notice.source.user_id is None
    assert store._generate_session_key(notice.source) == RECALL_PARTIAL_KEY

    await adapter.cancel_background_tasks()

    assert _payload_keys(flush_dir) == [RECALL_SLOT_KEY]
    recovered = recover_pending_to_db(db, session_resolver=store.resolve_session_id_for_key)
    assert recovered == 1
    assert [call["session_id"] for call in db.appended] == ["sid-recall"]
    assert [call["content"] for call in db.appended] == [notice.text]
    assert _spooled_files(flush_dir) == []


@pytest.mark.asyncio
async def test_the_runner_flush_keeps_a_routed_slot_key_too(tmp_path, flush_dir):
    """``runner.stop()`` keys its queue head the same way: routed slot key, not a re-derivation."""
    db = _FakeGatewayDB({RECALL_SLOT_KEY: {"id": "sid-recall", "started_at": 0.0}})
    runner, adapter, store = _wire_aligned(tmp_path, db)
    notice = _queue_recall_notice(adapter)
    adapter._pending_messages.pop(RECALL_SLOT_KEY)  # moved into the runner's queue head
    runner.session_store = store
    runner._pending_messages[RECALL_SLOT_KEY] = notice

    with patch("gateway.status.remove_pid_file"), patch(
        "gateway.status.publish_runtime_status"
    ):
        await runner.stop()

    assert _payload_keys(flush_dir) == [RECALL_SLOT_KEY]
    recovered = recover_pending_to_db(db, session_resolver=store.resolve_session_id_for_key)
    assert recovered == 1
    assert [call["session_id"] for call in db.appended] == ["sid-recall"]

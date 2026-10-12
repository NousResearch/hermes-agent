"""A sender-less Yuanbao recall notice must survive the shutdown flush into its own session.

Reported in review of this PR: a value whose ``source`` names no participant is queued by a
platform producer straight into the slot of the turn it belongs to, so a flush that re-derives
the key from that source alone loses the participant segment the slot carries and the notice
strands in the spool instead of replaying. The pre-PR flush recorded the slot key verbatim.

The recall guard is that producer: it rebuilds the notice's source from the recall callback, which
``DecodeMiddleware.parse_json_push`` admits without ``from_account``. This test drives the real
decode, the real recall producer, the real adapter shutdown flush and the real
SessionStore/SQLite reopen, and asserts the exact notice text lands in the ORIGINAL session with
the spool drained. ``from_account=""`` is the supported partial-source case; ``from_account="u1"``
is a positive control proving the fixture and the recovery seam are sound.
"""

import asyncio
import json
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.yuanbao import (
    DecodeMiddleware,
    InboundContext,
    RecallGuardMiddleware,
    YuanbaoAdapter,
)
from gateway.session import SessionStore
from gateway.shutdown_flush import recover_pending_to_db

RECALLED_MSG_ID = "m-original"
GROUP_CODE = "G"
ORIGINAL_SENDER = "u1"
# Both the adapter (PlatformConfig.extra) and the store use the gateway-wide isolation flags, so
# the slot key the notice is queued under IS the key the store routes the original turn by.
FLAGS = dict(group_sessions_per_user=True, thread_sessions_per_user=True)


def _config() -> GatewayConfig:
    return GatewayConfig(**FLAGS)


def _open_store(tmp_path: Path) -> SessionStore:
    return SessionStore(sessions_dir=tmp_path / "sessions", config=_config())


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Real SQLite under an isolated HERMES_HOME (same seam as the other reviewer probes)."""
    import hermes_constants
    import hermes_state

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    return tmp_path


def _wire(tmp_path: Path):
    store = _open_store(tmp_path)
    adapter = YuanbaoAdapter(PlatformConfig(enabled=True, token="***"))
    adapter.config.extra.update(**FLAGS)
    adapter.set_session_store(store)
    return store, adapter


def _original_source(adapter: YuanbaoAdapter):
    """The real group source of the message that is being processed when the recall arrives."""
    return adapter.build_source(
        chat_id=f"group:{GROUP_CODE}", chat_type="group",
        user_id=ORIGINAL_SENDER, thread_id="main",
    )


async def _noop() -> None:
    return None


async def _emit_recall(adapter: YuanbaoAdapter, from_account: str) -> str:
    """Drive the real decode + real recall producer; return the notice text queued for the slot."""
    raw = {
        "callback_command": "Group.CallbackAfterRecallMsg",
        "group_code": GROUP_CODE,
        "recall_msg_seq_list": [{"msg_id": RECALLED_MSG_ID}],
    }
    if from_account:  # otherwise exercise the supported callback without this key
        raw["from_account"] = from_account
    push = DecodeMiddleware.parse_json_push(raw)
    assert push is not None, "the decoder must admit this recall callback"
    assert push["callback_command"] == "Group.CallbackAfterRecallMsg"
    assert push["from_account"] == from_account

    slot = adapter._source_session_key(_original_source(adapter))
    adapter._processing_msg_ids[slot] = RECALLED_MSG_ID
    adapter._processing_msg_texts[slot] = ""  # blank: no content-redact poll
    adapter._active_sessions[slot] = asyncio.Event()

    await RecallGuardMiddleware().handle(InboundContext(adapter=adapter, push=push), _noop)

    notice = adapter._pending_messages.get(slot)
    assert notice is not None, "the recall interrupt must queue its notice on the matched slot"
    return notice.text


def _spool_keys(flush_dir: Path) -> list[str]:
    return sorted(json.loads(p.read_text(encoding="utf-8"))["session_key"]
                  for p in flush_dir.glob("pending-*.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("from_account", ["", ORIGINAL_SENDER])
async def test_recall_notice_replays_into_the_original_session(tmp_path, hermes_home, from_account):
    """The queued notice must survive a real adapter shutdown flush and land in the original row."""
    store, adapter = _wire(tmp_path)
    try:
        entry = store.get_or_create_session(_original_source(adapter))
        slot = adapter._source_session_key(_original_source(adapter))
        # Precondition: the slot key and the store key are the same string (flags agree).
        assert slot == entry.session_key == store._generate_session_key(_original_source(adapter))

        notice_text = await _emit_recall(adapter, from_account)

        # Real production flush seam: the exact changed line in base.py.
        await adapter.cancel_background_tasks()

        store.close_all_db_handles()
        reopened = _open_store(tmp_path)
        try:
            db = reopened._db_for_key(entry.session_key)
            recovered = recover_pending_to_db(
                db, session_resolver=reopened.resolve_session_id_for_key)
            contents = [m["content"] for m in db.get_messages(entry.session_id)]
        finally:
            reopened.close_all_db_handles()

        assert recovered == 1, (
            "the recall notice was not recovered; spool keys="
            f"{_spool_keys(tmp_path / 'pending_messages')}"
        )
        assert contents == [notice_text], f"notice text not in the original session: {contents!r}"
        assert not list((tmp_path / "pending_messages").glob("*.json")), "spool not drained"
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("from_account", ["", ORIGINAL_SENDER])
async def test_flushed_key_is_the_slot_the_notice_was_queued_under(tmp_path, hermes_home,
                                                                   from_account):
    """The spooled key must be the slot the notice was queued under, so recovery can resolve it."""
    store, adapter = _wire(tmp_path)
    try:
        store.get_or_create_session(_original_source(adapter))
        slot = adapter._source_session_key(_original_source(adapter))
        await _emit_recall(adapter, from_account)

        await adapter.cancel_background_tasks()

        assert _spool_keys(tmp_path / "pending_messages") == [slot]
    finally:
        store.close_all_db_handles()


# -- the DM branch of the same producer (C2C.CallbackAfterMsgWithDraw) -------------------------


def _original_dm_source(adapter: YuanbaoAdapter):
    """The real direct-chat source of the message being processed when the withdrawal arrives."""
    return adapter.build_source(
        chat_id=f"direct:{ORIGINAL_SENDER}", chat_type="dm", user_id=ORIGINAL_SENDER,
    )


async def _emit_c2c_recall(adapter: YuanbaoAdapter, from_account: str) -> str:
    """Drive the real decode + real C2C withdrawal producer; return the queued notice text."""
    raw = {
        "callback_command": "C2C.CallbackAfterMsgWithDraw",
        "msg_id": RECALLED_MSG_ID,
        "msg_seq": 7,
    }
    if from_account:
        raw["from_account"] = from_account
    push = DecodeMiddleware.parse_json_push(raw)
    assert push is not None, "the decoder must admit this withdrawal callback"

    slot = adapter._source_session_key(_original_dm_source(adapter))
    adapter._processing_msg_ids[slot] = RECALLED_MSG_ID
    adapter._processing_msg_texts[slot] = ""  # blank: no content-redact poll
    adapter._active_sessions[slot] = asyncio.Event()

    await RecallGuardMiddleware().handle(InboundContext(adapter=adapter, push=push), _noop)

    notice = adapter._pending_messages.get(slot)
    assert notice is not None, "the withdrawal interrupt must queue its notice on the matched slot"
    return notice.text


@pytest.mark.asyncio
@pytest.mark.parametrize("from_account", ["", ORIGINAL_SENDER])
async def test_c2c_withdrawal_notice_replays_into_the_original_dm_session(tmp_path, hermes_home,
                                                                         from_account):
    """A direct-chat notice carries the chat it interrupts, so a shutdown cannot move it."""
    store, adapter = _wire(tmp_path)
    try:
        entry = store.get_or_create_session(_original_dm_source(adapter))
        slot = adapter._source_session_key(_original_dm_source(adapter))
        assert slot == entry.session_key == store._generate_session_key(_original_dm_source(adapter))

        notice_text = await _emit_c2c_recall(adapter, from_account)
        assert _spool_keys(tmp_path / "pending_messages") == []

        await adapter.cancel_background_tasks()

        assert _spool_keys(tmp_path / "pending_messages") == [slot], (
            "the withdrawal notice was spooled under a key the store does not route by"
        )

        store.close_all_db_handles()
        reopened = _open_store(tmp_path)
        try:
            db = reopened._db_for_key(entry.session_key)
            recovered = recover_pending_to_db(
                db, session_resolver=reopened.resolve_session_id_for_key)
            contents = [m["content"] for m in db.get_messages(entry.session_id)]
        finally:
            reopened.close_all_db_handles()

        assert recovered == 1
        assert contents == [notice_text]
    finally:
        store.close_all_db_handles()

"""One Slack thread key is a single writer: group vs thread chat_type must not fork.

Observed 2026-09-16 on thread 1789519335.652199 (#proj-production-line):
inbound Slack stamps chat_type=\"group\" + thread_id, while a kanban wake can
stamp chat_type=\"thread\" + the same thread_id. ``build_session_key`` copies
``source.chat_type`` into the key whenever a real thread_id is present
(gateway/session.py ``chat_type_slot``), so those two sources publish two
live session_ids under:

  agent:main:slack:group:<team>:<channel>:<thread_ts>
  agent:main:slack:thread:<team>:<channel>:<thread_ts>

and both act. The claim path (get_or_create_session) must converge them.
"""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore, build_session_key


TEAM = "T0ALG8TE4JZ"
CHANNEL = "C0C26JFSENM"
THREAD = "1789519335.652199"
USER = "U0BV0T2479S"


def _db() -> MagicMock:
    db = MagicMock()
    db.get_session.return_value = None
    db.find_latest_gateway_session_for_peer.return_value = None
    db.reopen_session.return_value = None
    db.create_session.return_value = None
    db.get_compression_tip.side_effect = lambda sid: sid
    return db


def _make_store(tmp_path) -> tuple[SessionStore, MagicMock]:
    config = GatewayConfig()
    db = _db()
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = db
    store._loaded = True
    return store, db


def _source(*, chat_type: str) -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        chat_id=CHANNEL,
        chat_type=chat_type,
        user_id=USER,
        thread_id=THREAD,
        scope_id=TEAM,
    )


def test_slack_group_and_thread_chat_types_claim_one_session(tmp_path):
    """A group-stamped inbound and a thread-stamped wake for the same Slack
    thread must publish exactly one session_id — two claims, one writer."""
    store, db = _make_store(tmp_path)
    inbound = _source(chat_type="group")
    wake = _source(chat_type="thread")

    first = store.get_or_create_session(inbound)
    second = store.get_or_create_session(wake)

    assert first.session_id == second.session_id
    live_ids = {entry.session_id for entry in store._entries.values()}
    assert live_ids == {first.session_id}
    created_ids = {call.kwargs["session_id"] for call in db.create_session.call_args_list}
    assert created_ids == {first.session_id}


def test_concurrent_group_and_thread_claims_one_session_acts(tmp_path):
    """Overlapping claims on the same Slack thread (group vs thread stamp)
    still converge on one published session; both callers receive it."""
    store, db = _make_store(tmp_path)
    inbound = _source(chat_type="group")
    wake = _source(chat_type="thread")

    with ThreadPoolExecutor(max_workers=2) as pool:
        entries = [
            future.result(timeout=10)
            for future in (
                pool.submit(store.get_or_create_session, inbound),
                pool.submit(store.get_or_create_session, wake),
            )
        ]

    assert entries[0].session_id == entries[1].session_id
    live_ids = {entry.session_id for entry in store._entries.values()}
    assert live_ids == {entries[0].session_id}
    created_ids = {call.kwargs["session_id"] for call in db.create_session.call_args_list}
    assert created_ids == {entries[0].session_id}
    # Keys may alias, but they name one conversation — the thread_id is in both.
    assert THREAD in build_session_key(inbound)
    assert THREAD in build_session_key(wake)
    assert build_session_key(inbound) == build_session_key(wake)

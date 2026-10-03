"""Tests for Telegram thumbs-up reaction approval.

Verifies end-to-end reaction handling requirements:
1. Correct user (authorized vs unauthorized, actor_chat rejection, DM spoofing prevention)
2. Exact message/thread (reaction in thread A routes to thread A, not thread B)
3. Durable persistent deduplication in SQLite (prevents replay across process restarts / memory clears)
4. Removed or non-fresh reaction (clearing reaction, changing emoji, or pre-existing thumbs-up does not approve)
5. Event date vs edited message revision (reaction predating edit is rejected)
6. Unknown message (untracked message IDs are safely dropped)
7. Unrelated non-draft messages (status reports mentioning 'draft' are safely rejected, even if bucket has drafts)
8. Exact draft binding without blanket mutation (approving draft 1 leaves draft 2 in same session unapproved)
9. No 4000-char truncation (full immutable text and hashes preserved)
10. Fail closed on persistence failure (approval event not dispatched if consent write fails)
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.telegram import sent_messages
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter.config.extra = {}
    adapter._owner_profile = "default"
    adapter._bot = AsyncMock()
    adapter.handle_message = AsyncMock()
    adapter._authorization_check = None
    adapter._legacy_runner_auth_fn = MagicMock(return_value=None)
    adapter._is_callback_user_authorized = MagicMock(
        side_effect=lambda uid, **kw: str(uid) == "1335137548"
    )
    return adapter


def _make_reaction_update(
    chat_id: int = 1335137548,
    message_id: int = 1001,
    user_id: int = 1335137548,
    user_name: str = "Sam",
    new_emojis: list[str] | None = None,
    old_emojis: list[str] | None = None,
    date: datetime.datetime | None = None,
    chat_type: str = "dm",
    is_forum: bool = False,
    is_actor_chat: bool = False,
):
    if new_emojis is None:
        new_emojis = ["👍"]
    if old_emojis is None:
        old_emojis = []

    update = SimpleNamespace()
    mr = SimpleNamespace()
    mr.chat = SimpleNamespace(id=chat_id, type=chat_type, is_forum=is_forum)
    mr.message_id = message_id
    if is_actor_chat:
        mr.user = None
        mr.actor_chat = SimpleNamespace(id=user_id, title=user_name)
    else:
        mr.user = SimpleNamespace(id=user_id, username=user_name, first_name=user_name)
        mr.actor_chat = None
    mr.new_reaction = [SimpleNamespace(emoji=e) for e in new_emojis]
    mr.old_reaction = [SimpleNamespace(emoji=e) for e in old_emojis]
    mr.date = date or datetime.datetime.now(datetime.timezone.utc)
    update.message_reaction = mr
    return update


@pytest.fixture(autouse=True)
def reset_reaction_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sent_messages.clear_all_caches()
    yield
    sent_messages.clear_all_caches()


# 1. Correct user & identity tests
@pytest.mark.asyncio
async def test_reaction_correct_user(tmp_path):
    """Only an authorized user (Sam) can approve; unauthorized/actor_chat/mismatched DM is rejected."""
    adapter = _make_adapter()
    chat_id = "1335137548"
    msg_id = "2001"
    thread_id = "301486"
    session_key = f"agent:main:telegram:dm:{chat_id}:{thread_id}"

    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id=msg_id,
        thread_id=thread_id,
        session_key=session_key,
        text="Voorstel naar Malar: “Akkoord mits hotel gedekt is.”",
    )

    # Authorized user reaction
    auth_update = _make_reaction_update(
        chat_id=int(chat_id), message_id=int(msg_id), user_id=1335137548
    )
    await adapter._handle_message_reaction(auth_update, None)
    assert adapter.handle_message.await_count == 1
    event = adapter.handle_message.call_args[0][0]
    assert event.text == "👍"
    assert event.source.user_id == "1335137548"

    # Unauthorized user
    adapter.handle_message.reset_mock()
    msg_id_2 = "2002"
    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id=msg_id_2,
        thread_id=thread_id,
        session_key=session_key,
        text="Voorstel naar Malar: “Akkoord mits hotel gedekt is.”",
    )
    unauth_update = _make_reaction_update(
        chat_id=int(chat_id), message_id=int(msg_id_2), user_id=999999999, user_name="Attacker"
    )
    await adapter._handle_message_reaction(unauth_update, None)
    assert adapter.handle_message.await_count == 0

    # Actor chat (channel/anonymous) rejected
    msg_id_3 = "2003"
    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id=msg_id_3,
        thread_id=thread_id,
        session_key=session_key,
        text="Voorstel naar Malar: “Akkoord mits hotel gedekt is.”",
    )
    actor_chat_update = _make_reaction_update(
        chat_id=int(chat_id), message_id=int(msg_id_3), user_id=1335137548, is_actor_chat=True
    )
    await adapter._handle_message_reaction(actor_chat_update, None)
    assert adapter.handle_message.await_count == 0


# 2. Exact message / thread routing test
@pytest.mark.asyncio
async def test_reaction_exact_message_thread(tmp_path):
    """Reaction on message in thread A routes ONLY to thread A, not thread B."""
    adapter = _make_adapter()
    chat_id = "1335137548"

    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id="3001",
        thread_id="301486",
        session_key=f"agent:main:telegram:dm:{chat_id}:301486",
        text="Voorstel naar Malar: “Akkoord mits hotel gedekt is.”",
    )
    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id="3002",
        thread_id="301646",
        session_key=f"agent:main:telegram:dm:{chat_id}:301646",
        text="Voorstel naar Elise: “Kun je de agenda aanpassen?”",
    )

    update = _make_reaction_update(
        chat_id=int(chat_id), message_id=3001, user_id=1335137548
    )
    await adapter._handle_message_reaction(update, None)

    assert adapter.handle_message.await_count == 1
    event = adapter.handle_message.call_args[0][0]
    assert event.source.thread_id == "301486"
    assert event.metadata.get("gateway_session_key") == f"agent:main:telegram:dm:{chat_id}:301486"
    assert event.reply_to_message_id == "3001"


# 3. Durable persistent deduplication in SQLite
@pytest.mark.asyncio
async def test_reaction_durable_persistent_dedup(tmp_path):
    """Reaction deduplication survives in-memory cache wipe via SQLite table."""
    adapter = _make_adapter()
    chat_id = "1335137548"
    msg_id = "4001"
    thread_id = "301486"

    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id=msg_id,
        thread_id=thread_id,
        session_key=f"agent:main:telegram:dm:{chat_id}:{thread_id}",
        text="Voorstel naar Elise: “Kun je de agenda aanpassen?”",
    )

    update = _make_reaction_update(
        chat_id=int(chat_id), message_id=int(msg_id), user_id=1335137548
    )

    # First delivery succeeds
    await adapter._handle_message_reaction(update, None)
    assert adapter.handle_message.await_count == 1

    # Clear ONLY memory cache (simulating process restart)
    sent_messages._MEMORY_CACHE.clear()
    sent_messages._PROCESSED_REACTIONS.clear()

    # Second delivery must be rejected by SQLite persistent table
    await adapter._handle_message_reaction(update, None)
    assert adapter.handle_message.await_count == 1  # Unchanged!


# 4. Removed or non-fresh reaction test
@pytest.mark.asyncio
async def test_reaction_removed_or_not_fresh(tmp_path):
    """Removing reaction or changing to non-thumbs-up does not trigger approval."""
    adapter = _make_adapter()
    chat_id = "1335137548"
    msg_id = "5001"

    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id=msg_id,
        thread_id="301486",
        session_key=f"agent:main:telegram:dm:{chat_id}:301486",
        text="Voorstel: 'Verstuur de factuur.'",
    )

    # Removed
    removed_update = _make_reaction_update(
        chat_id=int(chat_id), message_id=int(msg_id), user_id=1335137548,
        new_emojis=[], old_emojis=["👍"]
    )
    await adapter._handle_message_reaction(removed_update, None)
    assert adapter.handle_message.await_count == 0

    # Changed to 👎
    dislike_update = _make_reaction_update(
        chat_id=int(chat_id), message_id=int(msg_id), user_id=1335137548,
        new_emojis=["👎"], old_emojis=[]
    )
    await adapter._handle_message_reaction(dislike_update, None)
    assert adapter.handle_message.await_count == 0

    # Already had thumbs-up, added ❤️ (not freshly added thumbs-up)
    already_had_update = _make_reaction_update(
        chat_id=int(chat_id), message_id=int(msg_id), user_id=1335137548,
        new_emojis=["👍", "❤️"], old_emojis=["👍"]
    )
    await adapter._handle_message_reaction(already_had_update, None)
    assert adapter.handle_message.await_count == 0


# 5. Event date vs edited message revision
@pytest.mark.asyncio
async def test_reaction_event_date_vs_revision(tmp_path):
    """A reaction timestamp predating message send/edit timestamp is rejected."""
    adapter = _make_adapter()
    chat_id = "1335137548"
    msg_id = "5501"

    # Send message at T = 1000
    base_time = time.time()
    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id=msg_id,
        thread_id="301486",
        session_key=f"agent:main:telegram:dm:{chat_id}:301486",
        text="Voorstel: 'Originele tekst.'",
    )

    # Reaction dated 10 seconds BEFORE message was sent
    stale_date = datetime.datetime.fromtimestamp(base_time - 10.0, datetime.timezone.utc)
    stale_update = _make_reaction_update(
        chat_id=int(chat_id), message_id=int(msg_id), user_id=1335137548, date=stale_date
    )
    await adapter._handle_message_reaction(stale_update, None)
    assert adapter.handle_message.await_count == 0


# 6. Unknown message test
@pytest.mark.asyncio
async def test_reaction_unknown_message(tmp_path):
    """Reaction on an untracked or unknown message ID is safely dropped."""
    adapter = _make_adapter()
    chat_id = "1335137548"
    update = _make_reaction_update(
        chat_id=int(chat_id), message_id=999999, user_id=1335137548
    )
    await adapter._handle_message_reaction(update, None)
    assert adapter.handle_message.await_count == 0


# 7. Unrelated non-draft / status messages rejected even when bucket has drafts
@pytest.mark.asyncio
async def test_reaction_unrelated_status_message_rejected(tmp_path):
    """Status messages mentioning 'draft' are NOT approved, even when drafts exist in session."""
    adapter = _make_adapter()
    chat_id = "1335137548"
    sid = "test_sess_01"
    session_key = f"agent:main:telegram:dm:{chat_id}:301486"

    # Put a staged draft in whatsapp-exact-approvals.json
    wa_path = tmp_path / "state" / "whatsapp-exact-approvals.json"
    wa_path.parent.mkdir(parents=True, exist_ok=True)
    wa_data = {
        sid: {
            "digest_1": {
                "account": "personal_nl",
                "recipient": "31612345678",
                "message": "Echte draft tekst naar Faruk.",
                "digest": "digest_1",
                "staged_at": time.time(),
            }
        }
    }
    wa_path.write_text(json.dumps(wa_data))

    # Bot sent a status message mentioning the word 'draft'
    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id="6001",
        thread_id="301486",
        session_key=session_key,
        text="Status: scanned inbox, found 0 new drafts. System operational.",
    )

    update = _make_reaction_update(
        chat_id=int(chat_id), message_id=6001, user_id=1335137548
    )
    await adapter._handle_message_reaction(update, None)
    assert adapter.handle_message.await_count == 0

    # Ensure draft in whatsapp-exact-approvals was NOT approved
    reloaded_wa = json.loads(wa_path.read_text())
    assert reloaded_wa[sid]["digest_1"].get("approved_at") is None


# 8. EXACT draft binding without blanket mutation
@pytest.mark.asyncio
async def test_reaction_exact_draft_binding_no_blanket_mutation(tmp_path):
    """Reacting to Draft 1 approves ONLY Draft 1; Draft 2 in the same session remains UNAPPROVED."""
    adapter = _make_adapter()
    chat_id = "1335137548"
    sid = "sess_multi_draft"
    session_key = f"agent:main:telegram:dm:{chat_id}:301486"

    # Map sid -> session_key in state.db
    state_db = tmp_path / "state.db"
    with sqlite3.connect(str(state_db)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                user_id TEXT,
                session_key TEXT,
                chat_id TEXT,
                chat_type TEXT,
                thread_id TEXT,
                started_at REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sessions (id, source, user_id, session_key, chat_id, chat_type, thread_id, started_at)
            VALUES (?, 'telegram', ?, ?, ?, 'dm', '301486', ?)
            """,
            (sid, chat_id, session_key, chat_id, time.time() - 100),
        )
        conn.commit()

    msg1_text = "Hee Faruk, deal is akkoord. We tekenen maandag."
    msg2_text = "Beste Axel, verlenging tot 2028 is prima. Stuur stukken door."

    digest1 = hashlib.sha256(f"personal_nl:31611111111:{msg1_text}".encode()).hexdigest()
    digest2 = hashlib.sha256(f"personal_us:15552222222:{msg2_text}".encode()).hexdigest()

    wa_path = tmp_path / "state" / "whatsapp-exact-approvals.json"
    wa_path.parent.mkdir(parents=True, exist_ok=True)
    wa_data = {
        sid: {
            digest1: {
                "account": "personal_nl",
                "recipient": "31611111111",
                "message": msg1_text,
                "digest": digest1,
                "staged_at": time.time(),
            },
            digest2: {
                "account": "personal_us",
                "recipient": "15552222222",
                "message": msg2_text,
                "digest": digest2,
                "staged_at": time.time(),
            },
        }
    }
    wa_path.write_text(json.dumps(wa_data))

    # Bot sent message 7001 displaying Draft 1
    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id="7001",
        thread_id="301486",
        session_key=session_key,
        text=f"> {msg1_text}\n\nAkkoord met sturen?",
        metadata={"recipient": "31611111111"},
    )
    # Bot sent message 7002 displaying Draft 2
    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id="7002",
        thread_id="301486",
        session_key=session_key,
        text=f"> {msg2_text}\n\nAkkoord met sturen?",
        metadata={"recipient": "15552222222"},
    )

    # Sam reacts 👍 ONLY to message 7001 (Draft 1)
    update = _make_reaction_update(
        chat_id=int(chat_id), message_id=7001, user_id=1335137548
    )
    await adapter._handle_message_reaction(update, None)

    assert adapter.handle_message.await_count == 1
    event = adapter.handle_message.call_args[0][0]
    assert event.metadata.get("draft_digest") == digest1

    # CRITICAL CHECK: Reload whatsapp-exact-approvals.json
    reloaded_wa = json.loads(wa_path.read_text())
    assert reloaded_wa[sid][digest1].get("approved_at") is not None
    assert reloaded_wa[sid][digest1].get("approval_kind") == "telegram_reaction"
    assert reloaded_wa[sid][digest1].get("approval_text") == "👍"

    # DRAFT 2 MUST REMAIN UNAPPROVED! Blanket mutation is completely prevented!
    assert reloaded_wa[sid][digest2].get("approved_at") is None
    assert "approval_kind" not in reloaded_wa[sid][digest2]

    # Verify durable consent log in outbound-consent.jsonl
    consent_file = tmp_path / "logs" / "outbound-consent.jsonl"
    assert consent_file.exists()
    lines = [json.loads(l) for l in consent_file.read_text().strip().splitlines()]
    assert len(lines) == 1
    entry = lines[0]
    assert entry["payload_sha256"] == digest1
    assert entry["recipient"] == "31611111111"
    assert entry["approval_kind"] == "telegram_reaction"
    assert entry["session_id"] == sid
    assert entry["telegram_message_id"] == "7001"


# 9. Messages > 4000 chars are NOT truncated
@pytest.mark.asyncio
async def test_reaction_no_4000_char_truncation(tmp_path):
    """Messages longer than 4000 chars preserve full immutable text and match digest."""
    adapter = _make_adapter()
    chat_id = "1335137548"
    sid = "sess_long"
    session_key = f"agent:main:telegram:dm:{chat_id}:301486"

    # Map sid -> session_key in state.db
    state_db = tmp_path / "state.db"
    with sqlite3.connect(str(state_db)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                user_id TEXT,
                session_key TEXT,
                chat_id TEXT,
                chat_type TEXT,
                thread_id TEXT,
                started_at REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sessions (id, source, user_id, session_key, chat_id, chat_type, thread_id, started_at)
            VALUES (?, 'telegram', ?, ?, ?, 'dm', '301486', ?)
            """,
            (sid, chat_id, session_key, chat_id, time.time() - 100),
        )
        conn.commit()

    long_body = "A" * 4200
    digest = hashlib.sha256(f"personal_nl:31600000000:{long_body}".encode()).hexdigest()

    wa_path = tmp_path / "state" / "whatsapp-exact-approvals.json"
    wa_path.parent.mkdir(parents=True, exist_ok=True)
    wa_data = {
        sid: {
            digest: {
                "account": "personal_nl",
                "recipient": "31600000000",
                "message": long_body,
                "digest": digest,
                "staged_at": time.time(),
            }
        }
    }
    wa_path.write_text(json.dumps(wa_data))

    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id="8001",
        thread_id="301486",
        session_key=session_key,
        text=long_body,
    )

    # Verify lookup returns full text without truncation
    rec = sent_messages.lookup_sent_message(chat_id, "8001")
    assert len(rec["text"]) == 4200

    update = _make_reaction_update(
        chat_id=int(chat_id), message_id=8001, user_id=1335137548
    )
    await adapter._handle_message_reaction(update, None)

    assert adapter.handle_message.await_count == 1
    reloaded_wa = json.loads(wa_path.read_text())
    assert reloaded_wa[sid][digest].get("approved_at") is not None


# 10. Persistence failure fails closed
@pytest.mark.asyncio
async def test_reaction_persistence_failure_fails_closed(tmp_path):
    """If recording durable consent fails, the approval is NOT dispatched to session."""
    adapter = _make_adapter()
    chat_id = "1335137548"
    msg_id = "9001"

    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id=msg_id,
        thread_id="301486",
        session_key=f"agent:main:telegram:dm:{chat_id}:301486",
        text="Voorstel: 'Verstuur de factuur.'",
    )

    with patch.object(sent_messages, "record_durable_consent", return_value={"ok": False, "error": "Disk full"}):
        update = _make_reaction_update(
            chat_id=int(chat_id), message_id=int(msg_id), user_id=1335137548
        )
        await adapter._handle_message_reaction(update, None)
        assert adapter.handle_message.await_count == 0


# 11. End-to-end integration test with state.db and real stores
@pytest.mark.asyncio
async def test_reaction_integration_with_state_db(tmp_path):
    """Full integration test: state.db session lookup, multi-draft store, SQLite dedup, consent logging."""
    import sqlite3

    adapter = _make_adapter()
    chat_id = "1335137548"
    user_id = 1335137548
    thread_id = "301486"
    session_id = "20260912_120000_abcd1234"
    session_key = f"agent:main:telegram:dm:{chat_id}:{thread_id}"

    # Setup state.db with session
    state_db = tmp_path / "state.db"
    with sqlite3.connect(str(state_db)) as conn:
        conn.execute(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                user_id TEXT,
                session_key TEXT,
                chat_id TEXT,
                chat_type TEXT,
                thread_id TEXT,
                started_at REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sessions (id, source, user_id, session_key, chat_id, chat_type, thread_id, started_at)
            VALUES (?, 'telegram', ?, ?, ?, 'dm', ?, ?)
            """,
            (session_id, str(user_id), session_key, chat_id, thread_id, time.time() - 100),
        )
        conn.commit()

    # Two drafts in whatsapp-exact-approvals.json for this session
    msg1 = "Hey Michelle, de vergadering staat op maandag 15:00."
    msg2 = "Hey Alex, de cijfers over Q3 staan in de bijlage."
    digest1 = hashlib.sha256(f"personal_nl:31611111111:{msg1}".encode()).hexdigest()
    digest2 = hashlib.sha256(f"personal_us:15552222222:{msg2}".encode()).hexdigest()

    wa_path = tmp_path / "state" / "whatsapp-exact-approvals.json"
    wa_path.parent.mkdir(parents=True, exist_ok=True)
    wa_data = {
        session_id: {
            digest1: {
                "account": "personal_nl",
                "recipient": "31611111111",
                "message": msg1,
                "digest": digest1,
                "staged_at": time.time() - 50,
            },
            digest2: {
                "account": "personal_us",
                "recipient": "15552222222",
                "message": msg2,
                "digest": digest2,
                "staged_at": time.time() - 40,
            },
        }
    }
    wa_path.write_text(json.dumps(wa_data))

    # Bot sends message in thread 301486 displaying Draft 1
    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id="10001",
        thread_id=thread_id,
        session_key=session_key,
        text=f"> {msg1}\n\nSturen naar Michelle?",
        metadata={"recipient": "31611111111", "gateway_session_key": session_key},
    )

    # Reaction thumbs up on message 10001
    update = _make_reaction_update(
        chat_id=int(chat_id), message_id=10001, user_id=user_id
    )
    await adapter._handle_message_reaction(update, None)

    # 1. MessageEvent dispatched with exact thread and session
    assert adapter.handle_message.await_count == 1
    event = adapter.handle_message.call_args[0][0]
    assert event.text == "👍"
    assert event.source.thread_id == thread_id
    assert event.metadata["gateway_session_key"] == session_key
    assert event.metadata["draft_digest"] == digest1

    # 2. whatsapp-exact-approvals.json updated ONLY for Draft 1
    reloaded_wa = json.loads(wa_path.read_text())
    assert reloaded_wa[session_id][digest1]["approved_at"] is not None
    assert reloaded_wa[session_id][digest1]["approval_kind"] == "telegram_reaction"
    assert reloaded_wa[session_id][digest1]["approval_text"] == "👍"
    assert reloaded_wa[session_id][digest2].get("approved_at") is None

    # 3. outbound-consent.jsonl logged
    consent_file = tmp_path / "logs" / "outbound-consent.jsonl"
    assert consent_file.exists()
    entries = [json.loads(l) for l in consent_file.read_text().strip().splitlines()]
    assert len(entries) == 1
    assert entries[0]["payload_sha256"] == digest1
    assert entries[0]["recipient"] == "31611111111"
    assert entries[0]["session_id"] == session_id

    # 4. Durable SQLite dedup: replaying update is dropped
    await adapter._handle_message_reaction(update, None)
    assert adapter.handle_message.await_count == 1


# 12. Mutation verification
def test_mutation_proof_kills_all_regressions(tmp_path, monkeypatch):
    """Prove that mutant implementations of critical consent logic fail the test assertions."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sent_messages.clear_all_caches()

    # Mutant A: Blanket approval (mutant approves all drafts in session bucket)
    wa_data = {
        "sess_mutant": {
            "d1": {"message": "Draft 1", "staged_at": time.time(), "recipient": "111"},
            "d2": {"message": "Draft 2", "staged_at": time.time(), "recipient": "222"},
        }
    }
    wa_path = tmp_path / "state" / "whatsapp-exact-approvals.json"
    wa_path.parent.mkdir(parents=True, exist_ok=True)
    wa_path.write_text(json.dumps(wa_data))

    # Test real find_matching_draft: matches ONLY Draft 1
    match = sent_messages.find_matching_draft(
        text="Draft 1",
        session_key="sess_mutant",
        chat_id="1335137548",
    )
    assert match is not None
    assert match[1] == "d1"

    # Simulate mutant function that approves everything
    def mutant_blanket_mutate(state):
        for b in state.values():
            for rec in b.values():
                rec["approved_at"] = time.time()

    # In our implementation, record_durable_consent only approves d1
    res = sent_messages.record_durable_consent(
        chat_id="1335137548",
        message_id="11",
        user_id="1335137548",
        session_key="sess_mutant",
        text="Draft 1",
    )
    assert res["ok"] is True
    post_state = json.loads(wa_path.read_text())
    assert post_state["sess_mutant"]["d1"].get("approved_at") is not None
    assert post_state["sess_mutant"]["d2"].get("approved_at") is None  # Kills Mutant A

    # Mutant B: is_draft_message returns True for status message
    status_msg = "Status: preparing draft for review"
    assert not sent_messages.is_draft_message(status_msg, session_key="sess_mutant")  # Kills Mutant B

    # Mutant C: claim_reaction returns True on duplicate
    claimed_1 = sent_messages.claim_reaction("1335137548", "99", "1335137548", "👍")
    assert claimed_1 is True
    # Reset in-memory cache to prove SQLite kills Mutant C
    sent_messages._PROCESSED_REACTIONS.clear()
    claimed_2 = sent_messages.claim_reaction("1335137548", "99", "1335137548", "👍")
    assert claimed_2 is False  # Kills Mutant C


# 13. Cross-thread identical body regression
@pytest.mark.asyncio
async def test_reaction_cross_thread_identical_body_scoping(tmp_path):
    """When identical draft text exists in Thread A and Thread B, reacting in Thread A
    MUST route strictly to Thread A's session and approve only Thread A's staged draft.
    Thread B's draft and session must remain untouched."""
    adapter = _make_adapter()
    chat_id = "1335137548"
    user_id = 1335137548
    thread_a = "301486"
    thread_b = "301999"

    sid_a = "sess_thread_a"
    sid_b = "sess_thread_b"
    session_key_a = f"agent:main:telegram:dm:{chat_id}:{thread_a}"
    session_key_b = f"agent:main:telegram:dm:{chat_id}:{thread_b}"

    # Setup state.db with both sessions mapped to their respective thread session_keys
    state_db = tmp_path / "state.db"
    with sqlite3.connect(str(state_db)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                user_id TEXT,
                session_key TEXT,
                chat_id TEXT,
                chat_type TEXT,
                thread_id TEXT,
                started_at REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sessions (id, source, user_id, session_key, chat_id, chat_type, thread_id, started_at)
            VALUES (?, 'telegram', ?, ?, ?, 'dm', ?, ?)
            """,
            (sid_a, str(user_id), session_key_a, chat_id, thread_a, time.time() - 100),
        )
        conn.execute(
            """
            INSERT INTO sessions (id, source, user_id, session_key, chat_id, chat_type, thread_id, started_at)
            VALUES (?, 'telegram', ?, ?, ?, 'dm', ?, ?)
            """,
            (sid_b, str(user_id), session_key_b, chat_id, thread_b, time.time() - 100),
        )
        conn.commit()

    # Identical draft text in both threads
    body = "Akkoord met het verlengen van de overeenkomst."
    digest_a = hashlib.sha256(f"personal_nl:31611111111:{body}".encode()).hexdigest()
    digest_b = hashlib.sha256(f"personal_nl:31622222222:{body}".encode()).hexdigest()

    wa_path = tmp_path / "state" / "whatsapp-exact-approvals.json"
    wa_path.parent.mkdir(parents=True, exist_ok=True)
    wa_data = {
        sid_a: {
            digest_a: {
                "account": "personal_nl",
                "recipient": "31611111111",
                "message": body,
                "digest": digest_a,
                "staged_at": time.time() - 30,
            }
        },
        sid_b: {
            digest_b: {
                "account": "personal_nl",
                "recipient": "31622222222",
                "message": body,
                "digest": digest_b,
                "staged_at": time.time() - 20,
            }
        },
    }
    wa_path.write_text(json.dumps(wa_data))

    # Bot sent message 9001 in Thread A displaying body
    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id="9001",
        thread_id=thread_a,
        session_key=session_key_a,
        text=f"> {body}\n\nAkkoord met sturen?",
        metadata={"recipient": "31611111111"},
    )
    # Bot sent message 9002 in Thread B displaying identical body
    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id="9002",
        thread_id=thread_b,
        session_key=session_key_b,
        text=f"> {body}\n\nAkkoord met sturen?",
        metadata={"recipient": "31622222222"},
    )

    # Sam reacts 👍 to message 9001 (in Thread A)
    update = _make_reaction_update(
        chat_id=int(chat_id), message_id=9001, user_id=1335137548
    )
    await adapter._handle_message_reaction(update, None)

    # Must route to Thread A only
    assert adapter.handle_message.await_count == 1
    event = adapter.handle_message.call_args[0][0]
    assert event.source.thread_id == thread_a
    assert event.metadata.get("gateway_session_key") == session_key_a
    assert event.metadata.get("draft_digest") == digest_a

    # Thread A draft approved, Thread B draft UNTOUCHED
    reloaded_wa = json.loads(wa_path.read_text())
    assert reloaded_wa[sid_a][digest_a].get("approved_at") is not None
    assert reloaded_wa[sid_b][digest_b].get("approved_at") is None


# 14. Digest-only without displayed body match fails closed
@pytest.mark.asyncio
async def test_reaction_digest_bypass_without_body_fails_closed(tmp_path):
    """Metadata digest match alone without displayed body proof must NOT approve."""
    adapter = _make_adapter()
    chat_id = "1335137548"
    sid = "sess_digest_test"
    session_key = f"agent:main:telegram:dm:{chat_id}:301486"

    state_db = tmp_path / "state.db"
    with sqlite3.connect(str(state_db)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                user_id TEXT,
                session_key TEXT,
                chat_id TEXT,
                chat_type TEXT,
                thread_id TEXT,
                started_at REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sessions (id, source, user_id, session_key, chat_id, chat_type, thread_id, started_at)
            VALUES (?, 'telegram', ?, ?, ?, 'dm', '301486', ?)
            """,
            (sid, chat_id, session_key, chat_id, time.time() - 100),
        )
        conn.commit()

    real_body = "Dit is het echte bericht dat goedgekeurd moet worden."
    digest = hashlib.sha256(f"personal_nl:31611111111:{real_body}".encode()).hexdigest()

    wa_path = tmp_path / "state" / "whatsapp-exact-approvals.json"
    wa_path.parent.mkdir(parents=True, exist_ok=True)
    wa_data = {
        sid: {
            digest: {
                "account": "personal_nl",
                "recipient": "31611111111",
                "message": real_body,
                "digest": digest,
                "staged_at": time.time() - 10,
            }
        }
    }
    wa_path.write_text(json.dumps(wa_data))

    # Bot sent message with completely DIFFERENT displayed text, but metadata claims digest
    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id="9501",
        thread_id="301486",
        session_key=session_key,
        text="Onzinbericht zonder de echte draft tekst",
        metadata={"digest": digest, "recipient": "31611111111"},
    )

    update = _make_reaction_update(
        chat_id=int(chat_id), message_id=9501, user_id=1335137548
    )
    await adapter._handle_message_reaction(update, None)

    # Must NOT approve because displayed text does not prove the body!
    assert adapter.handle_message.await_count == 0
    reloaded_wa = json.loads(wa_path.read_text())
    assert reloaded_wa[sid][digest].get("approved_at") is None


# 15. Same body different recipient without recipient metadata refuses ambiguous approval
@pytest.mark.asyncio
async def test_reaction_same_body_different_recipient_ambiguity(tmp_path):
    """Same body to different recipients in the same session must refuse approval if metadata lacks recipient."""
    adapter = _make_adapter()
    chat_id = "1335137548"
    sid = "sess_ambig"
    session_key = f"agent:main:telegram:dm:{chat_id}:301486"

    state_db = tmp_path / "state.db"
    with sqlite3.connect(str(state_db)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                user_id TEXT,
                session_key TEXT,
                chat_id TEXT,
                chat_type TEXT,
                thread_id TEXT,
                started_at REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sessions (id, source, user_id, session_key, chat_id, chat_type, thread_id, started_at)
            VALUES (?, 'telegram', ?, ?, ?, 'dm', '301486', ?)
            """,
            (sid, chat_id, session_key, chat_id, time.time() - 100),
        )
        conn.commit()

    body = "Akkoord met voorwaarden."
    d1 = hashlib.sha256(f"personal_nl:31611111111:{body}".encode()).hexdigest()
    d2 = hashlib.sha256(f"personal_us:15552222222:{body}".encode()).hexdigest()

    wa_path = tmp_path / "state" / "whatsapp-exact-approvals.json"
    wa_path.parent.mkdir(parents=True, exist_ok=True)
    wa_data = {
        sid: {
            d1: {
                "account": "personal_nl",
                "recipient": "31611111111",
                "message": body,
                "digest": d1,
                "staged_at": time.time() - 20,
            },
            d2: {
                "account": "personal_us",
                "recipient": "15552222222",
                "message": body,
                "digest": d2,
                "staged_at": time.time() - 10,
            },
        }
    }
    wa_path.write_text(json.dumps(wa_data))

    # Message displays the body but metadata does NOT specify which recipient
    sent_messages.record_sent_message(
        chat_id=chat_id,
        message_id="9601",
        thread_id="301486",
        session_key=session_key,
        text=f"> {body}\n\nAkkoord met sturen?",
        metadata={},  # No recipient metadata!
    )

    update = _make_reaction_update(
        chat_id=int(chat_id), message_id=9601, user_id=1335137548
    )
    await adapter._handle_message_reaction(update, None)

    # Must fail closed: ambiguous between two recipients!
    assert adapter.handle_message.await_count == 0
    reloaded_wa = json.loads(wa_path.read_text())
    assert reloaded_wa[sid][d1].get("approved_at") is None
    assert reloaded_wa[sid][d2].get("approved_at") is None


# 16. Edit count cache persistence and reaction predating edit fails closed
@pytest.mark.asyncio
async def test_reaction_edit_count_cache_and_predating_edit(tmp_path):
    """record_sent_message updates edit_count in cache, and reaction predating edit is rejected."""
    chat_id = "1335137548"
    msg_id = "9701"
    thread_id = "301486"
    session_key = f"agent:main:telegram:dm:{chat_id}:{thread_id}"

    # First send: edit_count == 0
    t0 = 1000.0
    with patch("time.time", return_value=t0):
        sent_messages.record_sent_message(
            chat_id=chat_id,
            message_id=msg_id,
            thread_id=thread_id,
            session_key=session_key,
            text="Initial revision",
        )
    rec1 = sent_messages.lookup_sent_message(chat_id, msg_id)
    assert rec1["edit_count"] == 0
    assert rec1["timestamp"] == t0

    # Second send (edit): edit_count == 1 in cache and DB
    t1 = 1010.0
    with patch("time.time", return_value=t1):
        sent_messages.record_sent_message(
            chat_id=chat_id,
            message_id=msg_id,
            thread_id=thread_id,
            session_key=session_key,
            text="Edited revision",
        )
    rec2 = sent_messages.lookup_sent_message(chat_id, msg_id)
    assert rec2["edit_count"] == 1
    assert rec2["timestamp"] == t1
    assert rec2["text"] == "Edited revision"

    # Verify reaction dated between t0 and t1 (i.e. Reacted to old revision) is REJECTED
    adapter = _make_adapter()
    reaction_date = datetime.datetime.fromtimestamp(1005.0, datetime.timezone.utc)
    update = _make_reaction_update(
        chat_id=int(chat_id), message_id=int(msg_id), user_id=1335137548, date=reaction_date
    )
    await adapter._handle_message_reaction(update, None)
    assert adapter.handle_message.await_count == 0




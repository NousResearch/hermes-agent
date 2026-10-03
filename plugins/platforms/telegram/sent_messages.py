"""Index and tracking for bot-sent Telegram messages and durable reaction approval consent.

Telegram message_reaction updates do NOT carry message_thread_id.
To support thumbs-up reaction approvals on specific assistant messages in specific
threads without bleeding into other threads, we track bot-sent messages
(chat_id, message_id) -> {thread_id, session_key, text, metadata, timestamp}.

Enforces exact draft binding with whatsapp-exact-approvals.json, durable persistent
deduplication in SQLite, revision ordering, and fail-closed persistence.
"""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Max messages kept in SQLite store
_MAX_STORED_MESSAGES = 5000

# Retention window for reaction approval (24 hours)
_RETENTION_SECONDS = 86400

# Window for draft validity in whatsapp-exact-approvals (12 hours)
APPROVAL_TTL_SECONDS = 12 * 60 * 60

# Lock timeouts
_LOCK_TIMEOUT_SECONDS = 10.0
_LOCK_POLL_INTERVAL_SECONDS = 0.05

# In-memory lookup cache: (chat_id, message_id) -> record
_MEMORY_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
# In-memory fast dedup cache: event_key -> bool
_PROCESSED_REACTIONS: Set[str] = set()

_DB_INITIALIZED: Set[str] = set()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS telegram_sent_messages (
    chat_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    thread_id TEXT,
    session_key TEXT,
    text TEXT,
    text_sha256 TEXT,
    metadata_json TEXT,
    timestamp REAL NOT NULL,
    edit_count INTEGER DEFAULT 0,
    PRIMARY KEY (chat_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_tsm_timestamp ON telegram_sent_messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_tsm_session ON telegram_sent_messages(session_key);

CREATE TABLE IF NOT EXISTS telegram_processed_reactions (
    chat_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    emoji TEXT NOT NULL,
    thread_id TEXT,
    session_key TEXT,
    processed_at REAL NOT NULL,
    PRIMARY KEY (chat_id, message_id, user_id, emoji)
);
CREATE INDEX IF NOT EXISTS idx_tpr_processed_at ON telegram_processed_reactions(processed_at);
"""


def _hermes_home() -> Path:
    env_home = os.getenv("HERMES_HOME")
    if env_home:
        return Path(env_home)
    return Path.home() / ".hermes"


def _db_path() -> Path:
    return _hermes_home() / "state" / "telegram_sent_messages.db"


def _consent_log_path() -> Path:
    return _hermes_home() / "logs" / "outbound-consent.jsonl"


def _init_db(conn: sqlite3.Connection) -> None:
    db_key = str(_db_path())
    if db_key in _DB_INITIALIZED:
        return
    conn.executescript(_SCHEMA)
    cursor = conn.execute("PRAGMA table_info(telegram_sent_messages);")
    columns = {row[1] for row in cursor.fetchall()}
    if "text_sha256" not in columns:
        try:
            conn.execute("ALTER TABLE telegram_sent_messages ADD COLUMN text_sha256 TEXT;")
        except Exception:
            pass
    if "edit_count" not in columns:
        try:
            conn.execute("ALTER TABLE telegram_sent_messages ADD COLUMN edit_count INTEGER DEFAULT 0;")
        except Exception:
            pass
    conn.commit()
    _DB_INITIALIZED.add(db_key)


def _connect_db() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    try:
        from hermes_state_wal import apply_wal_with_fallback
        apply_wal_with_fallback(conn, db_label="state/telegram_sent_messages.db")
    except Exception:
        pass
    conn.execute("PRAGMA busy_timeout=5000;")
    _init_db(conn)
    return conn


def clear_all_caches() -> None:
    """Clear memory caches and reset database init state (for tests)."""
    _MEMORY_CACHE.clear()
    _PROCESSED_REACTIONS.clear()
    _DB_INITIALIZED.clear()


def _dequote(text: str) -> str:
    out = []
    for line in str(text or "").split("\n"):
        stripped = line.lstrip()
        while stripped.startswith(">"):
            stripped = stripped[1:]
            if stripped.startswith(" "):
                stripped = stripped[1:]
        out.append(stripped if stripped != line.lstrip() else line)
    return "\n".join(out)


def _shows_body(haystack: str, message: str) -> bool:
    hay = str(haystack or "")
    msg = str(message or "")
    if not msg or not hay:
        return False
    return msg in hay or msg in _dequote(hay)


def _normalized_recipient(value: Any) -> str:
    raw = str(value or "").strip()
    if "@" in raw:
        return raw.lower()
    digits = re.sub(r"\D", "", raw)
    return digits or raw.lower()


def _acquire_lock_bounded(lock_file_obj, timeout: float = _LOCK_TIMEOUT_SECONDS) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(lock_file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except (BlockingIOError, OSError):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"could not acquire approval lock within {timeout}s")
            time.sleep(_LOCK_POLL_INTERVAL_SECONDS)


def _locked_approval_state(mutator):
    wa_path = _hermes_home() / "state" / "whatsapp-exact-approvals.json"
    lock_path = _hermes_home() / "state" / "whatsapp-exact-approvals.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        _acquire_lock_bounded(lock_file)
        try:
            try:
                state = json.loads(wa_path.read_text(encoding="utf-8")) if wa_path.exists() else {}
            except Exception:
                state = {}
            if not isinstance(state, dict):
                state = {}
            result = mutator(state)
            tmp = wa_path.with_suffix(f".tmp.{os.getpid()}")
            tmp.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            os.chmod(tmp, 0o600)
            tmp.replace(wa_path)
            return result
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _append_consent_log(entry: Dict[str, Any]) -> None:
    log_path = _consent_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        fh.flush()


def record_sent_message(
    chat_id: Any,
    message_id: Any,
    thread_id: Optional[Any] = None,
    session_key: Optional[str] = None,
    text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a bot-sent message to map (chat_id, message_id) -> thread_id and full displayed text.
    
    Stores the full immutable displayed text without 4000-character truncation.
    Tracks edit count and timestamps on revision updates.
    """
    if chat_id is None or message_id is None:
        return

    c_id = str(chat_id).strip()
    m_id = str(message_id).strip()
    t_id = str(thread_id).strip() if thread_id is not None else None
    s_key = str(session_key).strip() if session_key else None
    full_text = str(text or "")
    text_sha256 = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    now = time.time()

    meta_json = json.dumps(metadata or {}, ensure_ascii=False)

    old_rec = _MEMORY_CACHE.get((c_id, m_id))
    edit_count = (old_rec.get("edit_count", 0) + 1) if old_rec else 0

    record = {
        "chat_id": c_id,
        "message_id": m_id,
        "thread_id": t_id,
        "session_key": s_key,
        "text": full_text,
        "text_sha256": text_sha256,
        "metadata": metadata or {},
        "timestamp": now,
        "edit_count": edit_count,
    }

    _MEMORY_CACHE[(c_id, m_id)] = record

    try:
        with _connect_db() as conn:
            cursor = conn.execute(
                """
                INSERT INTO telegram_sent_messages
                    (chat_id, message_id, thread_id, session_key, text, text_sha256, metadata_json, timestamp, edit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(chat_id, message_id) DO UPDATE SET
                    thread_id=COALESCE(excluded.thread_id, telegram_sent_messages.thread_id),
                    session_key=COALESCE(excluded.session_key, telegram_sent_messages.session_key),
                    text=excluded.text,
                    text_sha256=excluded.text_sha256,
                    metadata_json=excluded.metadata_json,
                    timestamp=excluded.timestamp,
                    edit_count=telegram_sent_messages.edit_count + 1
                RETURNING edit_count
                """,
                (c_id, m_id, t_id, s_key, full_text, text_sha256, meta_json, now),
            )
            row = cursor.fetchone()
            if row:
                record["edit_count"] = int(row[0] or 0)
            conn.commit()
    except Exception as exc:
        logger.error("Failed to persist sent message %s in chat %s to DB: %s", m_id, c_id, exc, exc_info=True)
        raise


def lookup_sent_message(chat_id: Any, message_id: Any) -> Optional[Dict[str, Any]]:
    """Look up a bot-sent message by chat_id and message_id."""
    if chat_id is None or message_id is None:
        return None

    c_id = str(chat_id).strip()
    m_id = str(message_id).strip()
    key = (c_id, m_id)

    if key in _MEMORY_CACHE:
        return _MEMORY_CACHE[key]

    try:
        with _connect_db() as conn:
            cursor = conn.execute(
                """
                SELECT chat_id, message_id, thread_id, session_key, text, text_sha256, metadata_json, timestamp, edit_count
                FROM telegram_sent_messages
                WHERE chat_id=? AND message_id=?
                """,
                (c_id, m_id),
            )
            row = cursor.fetchone()
            if row:
                meta = {}
                try:
                    if row[6]:
                        meta = json.loads(row[6])
                except Exception:
                    pass
                rec = {
                    "chat_id": row[0],
                    "message_id": row[1],
                    "thread_id": row[2],
                    "session_key": row[3],
                    "text": row[4] or "",
                    "text_sha256": row[5] or "",
                    "metadata": meta,
                    "timestamp": float(row[7]),
                    "edit_count": int(row[8] or 0),
                }
                _MEMORY_CACHE[key] = rec
                return rec
    except Exception as exc:
        logger.error("Error looking up sent message %s in chat %s: %s", m_id, c_id, exc, exc_info=True)
    return None


def claim_reaction(
    chat_id: Any,
    message_id: Any,
    user_id: Any,
    emoji: str = "👍",
    thread_id: Optional[Any] = None,
    session_key: Optional[str] = None,
) -> bool:
    """Atomically claim a reaction event in SQLite.
    
    Returns True if successfully claimed (first time).
    Returns False if already claimed / duplicate replay.
    """
    c_id = str(chat_id).strip()
    m_id = str(message_id).strip()
    u_id = str(user_id).strip()
    em = str(emoji).strip()
    event_key = f"{c_id}:{m_id}:{u_id}:{em}"

    if event_key in _PROCESSED_REACTIONS:
        return False

    now = time.time()
    try:
        with _connect_db() as conn:
            conn.execute(
                """
                INSERT INTO telegram_processed_reactions
                    (chat_id, message_id, user_id, emoji, thread_id, session_key, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (c_id, m_id, u_id, em, str(thread_id) if thread_id is not None else None, session_key, now),
            )
            conn.commit()
            _PROCESSED_REACTIONS.add(event_key)
            return True
    except sqlite3.IntegrityError:
        _PROCESSED_REACTIONS.add(event_key)
        return False
    except Exception as exc:
        logger.error("Failed to claim reaction in SQLite: %s", exc, exc_info=True)
        return False


def is_reaction_already_processed(
    chat_id: Any, message_id: Any, user_id: Any, emoji: str = "👍"
) -> bool:
    """Return True if this reaction event was already processed."""
    c_id = str(chat_id).strip()
    m_id = str(message_id).strip()
    u_id = str(user_id).strip()
    em = str(emoji).strip()
    event_key = f"{c_id}:{m_id}:{u_id}:{em}"
    if event_key in _PROCESSED_REACTIONS:
        return True
    try:
        with _connect_db() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM telegram_processed_reactions WHERE chat_id=? AND message_id=? AND user_id=? AND emoji=?",
                (c_id, m_id, u_id, em),
            )
            if cursor.fetchone() is not None:
                _PROCESSED_REACTIONS.add(event_key)
                return True
    except Exception as exc:
        logger.error("Failed to check reaction in SQLite: %s", exc, exc_info=True)
    return False


def mark_reaction_processed(
    chat_id: Any, message_id: Any, user_id: Any, emoji: str = "👍",
    thread_id: Optional[Any] = None, session_key: Optional[str] = None,
) -> None:
    """Mark a reaction event as processed."""
    claim_reaction(chat_id, message_id, user_id, emoji, thread_id=thread_id, session_key=session_key)


def find_matching_draft(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    session_key: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Find the exact unapproved staged draft in whatsapp-exact-approvals.json matching this message.
    
    Binds by full immutable displayed bytes/hash and recipient.
    Returns (session_id, digest, draft_record) if an unambiguous match is found.
    Returns None if no draft matches or if multiple distinct drafts match.
    """
    clean_text = str(text or "").strip()
    if not clean_text and not metadata:
        return None

    wa_path = _hermes_home() / "state" / "whatsapp-exact-approvals.json"
    if not wa_path.exists():
        return None

    now = time.time()

    try:
        data = json.loads(wa_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
    except Exception as exc:
        logger.error("Failed to read whatsapp-exact-approvals.json: %s", exc, exc_info=True)
        return None

    candidate_sids: Set[str] = set()
    if session_key:
        candidate_sids.add(str(session_key).strip())

    state_db_path = _hermes_home() / "state.db"
    if state_db_path.exists():
        try:
            with sqlite3.connect(f"file:{state_db_path}?mode=ro", uri=True, timeout=1.0) as sdb:
                if session_key:
                    rows = sdb.execute(
                        "SELECT id FROM sessions WHERE session_key=? OR id=? ORDER BY started_at DESC LIMIT 10",
                        (session_key, session_key),
                    ).fetchall()
                    for r in rows:
                        if r and r[0]:
                            candidate_sids.add(str(r[0]))
                # Never widen approval scope to every session in the same chat.
        except Exception as exc:
            logger.debug("Failed querying state.db for candidate sessions: %s", exc)

    meta_digest = (metadata or {}).get("digest") or (metadata or {}).get("payload_sha256") or (metadata or {}).get("draft_digest")
    meta_recipient = (metadata or {}).get("recipient")

    matched_candidates: List[Tuple[str, str, Dict[str, Any]]] = []

    for sid, bucket in data.items():
        if not isinstance(bucket, dict):
            continue
        for digest, rec in bucket.items():
            if not isinstance(rec, dict):
                continue
            # 1. Freshness check
            staged_at = float(rec.get("staged_at") or 0.0)
            if staged_at > 0 and (now - staged_at > APPROVAL_TTL_SECONDS):
                continue
            # 2. Already approved?
            if rec.get("approved_at"):
                continue
            # 3. Recipient check if metadata specifies recipient
            if meta_recipient:
                rec_recipient = rec.get("recipient")
                if rec_recipient and _normalized_recipient(meta_recipient) != _normalized_recipient(rec_recipient):
                    continue
            # 4. Content match against displayed text (MANDATORY proof)
            draft_msg = str(rec.get("message") or "").strip()
            if not draft_msg or len(draft_msg) < 5:
                continue
            if not clean_text:
                continue

            has_body_proof = (
                clean_text == draft_msg
                or _dequote(clean_text).strip() == draft_msg
                or _shows_body(clean_text, draft_msg)
            )
            if not has_body_proof:
                continue

            # 5. Digest check if metadata specifies digest (must corroborate body proof)
            if meta_digest and meta_digest != digest:
                continue

            matched_candidates.append((sid, digest, rec))

    if not matched_candidates:
        return None

    in_session = [m for m in matched_candidates if m[0] in candidate_sids]
    working_set = in_session
    if not working_set:
        return None

    # Verify recipient disambiguation: same body to different recipients must not pick arbitrarily!
    distinct_recipients = {_normalized_recipient(m[2].get("recipient")) for m in working_set}
    if len(distinct_recipients) > 1 and not meta_recipient:
        logger.warning(
            "Multiple drafts with same body but different recipients matched without explicit recipient metadata: %s",
            distinct_recipients,
        )
        return None

    distinct_digests = {m[1] for m in working_set}
    if len(distinct_digests) == 1:
        return max(working_set, key=lambda m: float(m[2].get("staged_at") or 0.0))
    else:
        logger.warning(
            "Multiple distinct drafts matched reaction text; refusing ambiguous approval to prevent approving wrong draft: %s",
            distinct_digests,
        )
        return None


def is_draft_message(
    text: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
    session_key: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> bool:
    """Return True if text or metadata indicates an actionable draft awaiting approval.
    
    Rejects status messages, progress updates, and non-draft chatter even if
    they contain words like 'draft' or 'proposal'.
    Never marks a message as a draft merely because another draft is staged in the session.
    """
    if not text and not metadata:
        return False

    meta = metadata or {}
    if meta.get("is_status") or meta.get("status_key") or meta.get("is_ack"):
        return False

    clean_text = (text or "").strip()

    # Reject obvious status and progress reports
    if clean_text:
        if re.match(r"^(?:status|progress|voortgang|update|checking|bezig|done|klaar|error|fout)\b", clean_text, re.IGNORECASE):
            return False

    # Check explicit draft metadata
    if meta.get("is_draft") or meta.get("requires_approval") or meta.get("is_approval_prompt"):
        return True

    # Check if this exact text matches a staged draft in whatsapp-exact-approvals
    if clean_text:
        matched = find_matching_draft(clean_text, meta, session_key, chat_id)
        if matched is not None:
            return True

    # Check structured draft patterns in displayed text
    if clean_text:
        # Blockquote accompanied by explicit approval prompt
        has_blockquote = bool(re.search(r"^\s*>\s+\S+", clean_text, re.MULTILINE))
        has_approval_prompt = bool(re.search(r"\b(?:akkoord\?|sturen\?|send\?|shall i send|agree\?|tik\s+op\s+👍|react\s+with\s+👍)\b", clean_text, re.IGNORECASE))
        if has_blockquote and has_approval_prompt:
            return True

        # Explicit proposal header with quoted body: "Voorstel naar X: “...”" or "Proposal to Y: “...”"
        if re.search(r"^(?:voorstel|proposal|concept)[ \t]*(?:naar|to|voor)?[ \t]*[^:\n]{2,40}:[ \t]*[\"“'>]", clean_text, re.IGNORECASE | re.MULTILINE):
            return True

    return False


def record_durable_consent(
    chat_id: Any,
    message_id: Any,
    user_id: Any,
    thread_id: Optional[Any] = None,
    session_key: Optional[str] = None,
    text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record durable consent for the exact displayed draft message.
    
    Never blanket mutates all drafts. Binds only to the exact matching draft
    in whatsapp-exact-approvals.json or exact local draft text, writing an
    immutable log entry to outbound-consent.jsonl.
    """
    now = time.time()
    iso_ts = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).isoformat()
    clean_text = str(text or "").strip()
    meta = metadata or {}

    # Search for exact matching staged draft
    matched = find_matching_draft(clean_text, meta, session_key, chat_id)

    if matched is not None:
        target_sid, target_digest, draft_rec = matched

        def update_wa_state(state: Dict[str, Any]) -> bool:
            bucket = state.get(target_sid)
            if not isinstance(bucket, dict):
                return False
            rec = bucket.get(target_digest)
            if not isinstance(rec, dict):
                return False
            # Recheck freshness and unapproved status under shared lock
            staged_at = float(rec.get("staged_at") or 0.0)
            if staged_at > 0 and (now - staged_at > APPROVAL_TTL_SECONDS):
                return False
            if rec.get("approved_at"):
                return False
            # Recheck exact recipient and body before marking
            expected_recipient = draft_rec.get("recipient")
            if expected_recipient and _normalized_recipient(rec.get("recipient")) != _normalized_recipient(expected_recipient):
                return False
            expected_msg = str(draft_rec.get("message") or "")
            if str(rec.get("message") or "") != expected_msg:
                return False

            rec["approved_at"] = now
            rec["approval_kind"] = "telegram_reaction"
            rec["approval_text"] = "👍"
            rec["approval_event_id"] = int(message_id) if str(message_id).isdigit() else 0
            rec["approval_user_id"] = str(user_id)
            rec["approval_chat_id"] = str(chat_id)
            rec["approval_thread_id"] = str(thread_id) if thread_id is not None else None
            return True

        try:
            mutated = _locked_approval_state(update_wa_state)
            if not mutated:
                logger.error("Failed to mutate target draft %s in session %s (disappeared under lock)", target_digest, target_sid)
                return {"ok": False, "error": "Target draft disappeared during approval"}
        except Exception as exc:
            logger.error("Failed to update whatsapp-exact-approvals.json under lock: %s", exc, exc_info=True)
            return {"ok": False, "error": str(exc)}

        msg_str = str(draft_rec.get("message") or "")
        entry = {
            "ts": iso_ts,
            "channel": draft_rec.get("channel") or "whatsapp",
            "account": str(draft_rec.get("account") or ""),
            "recipient": str(draft_rec.get("recipient") or ""),
            "payload_sha256": target_digest,
            "full_message_sha256": hashlib.sha256(msg_str.encode("utf-8")).hexdigest(),
            "message_preview": msg_str[:160],
            "message_chars": len(msg_str),
            "approval_kind": "telegram_reaction",
            "approval_text": "👍",
            "approval_event_id": str(message_id),
            "session_id": target_sid,
            "telegram_chat_id": str(chat_id),
            "telegram_message_id": str(message_id),
            "telegram_user_id": str(user_id),
            "telegram_thread_id": str(thread_id) if thread_id is not None else None,
            "session_key": str(session_key or ""),
        }
        try:
            _append_consent_log(entry)
        except Exception as exc:
            logger.error("Failed to write to outbound-consent.jsonl: %s", exc, exc_info=True)
            return {"ok": False, "error": f"Failed writing consent log: {exc}"}

        return {"ok": True, "bound_draft": draft_rec, "digest": target_digest, "session_id": target_sid}

    # Not in whatsapp-exact-approvals: check if it's an explicit local draft
    is_local_draft = bool(meta.get("is_draft") or meta.get("requires_approval") or meta.get("is_proposal"))
    if not is_local_draft:
        if re.search(r"^(?:voorstel|proposal|concept)[ \t]*(?:naar|to|voor)?[ \t]*[^:\n]{2,40}:[ \t]*[\"“'>]", clean_text, re.IGNORECASE | re.MULTILINE):
            is_local_draft = True
        elif bool(re.search(r"^\s*>\s+\S+", clean_text, re.MULTILINE)) and bool(re.search(r"\b(?:akkoord\?|sturen\?|send\?|shall i send|agree\?|tik\s+op\s+👍|react\s+with\s+👍)\b", clean_text, re.IGNORECASE)):
            is_local_draft = True

    if is_local_draft and clean_text:
        text_sha = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        entry = {
            "ts": iso_ts,
            "channel": "telegram",
            "account": "",
            "recipient": str(meta.get("recipient") or ""),
            "payload_sha256": text_sha,
            "full_message_sha256": text_sha,
            "message_preview": clean_text[:160],
            "message_chars": len(clean_text),
            "approval_kind": "telegram_reaction",
            "approval_text": "👍",
            "approval_event_id": str(message_id),
            "session_id": str(session_key or ""),
            "telegram_chat_id": str(chat_id),
            "telegram_message_id": str(message_id),
            "telegram_user_id": str(user_id),
            "telegram_thread_id": str(thread_id) if thread_id is not None else None,
            "session_key": str(session_key or ""),
        }
        try:
            _append_consent_log(entry)
        except Exception as exc:
            logger.error("Failed to write to outbound-consent.jsonl: %s", exc, exc_info=True)
            return {"ok": False, "error": f"Failed writing consent log: {exc}"}

        return {"ok": True, "bound_draft": None, "digest": text_sha, "session_id": session_key or ""}

    logger.warning("No matching staged or local draft found for message %s in chat %s", message_id, chat_id)
    return {"ok": False, "error": "No matching draft found for message"}

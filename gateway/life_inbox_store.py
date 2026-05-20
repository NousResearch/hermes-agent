"""Account-scoped life inbox storage for gateway message ingestion.

Telegram Business/Profile Automation messages are stored in the owner's
account-scoped SQLite DB. Raw private chat text is kept out of gateway logs and
probe tables, but the main Business inbox archive stores plaintext messages so
the passive analyzer can summarize and retrieve them later.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PLATFORM_TELEGRAM_BUSINESS = "telegram_business"
DEFAULT_CHAT_RULE_MODE = "full_rag_selected"
SCHEMA_VERSION = 3
BUSINESS_PAYLOAD_PROBE_LANE = "business_bot_probe"
BUSINESS_PAYLOAD_PROBE_SCENARIOS: tuple[dict[str, str], ...] = (
    {
        "scenario_id": "S1_contact_inbound",
        "alias": "CONTACT_1",
        "expected_direction": "incoming_to_owner",
    },
    {
        "scenario_id": "S2_contact_alen_manual_outbound",
        "alias": "CONTACT_1",
        "expected_direction": "outgoing_from_owner",
    },
    {
        "scenario_id": "S3_known_noncontact_inbound",
        "alias": "KNOWN_NONCONTACT_1",
        "expected_direction": "incoming_to_owner",
    },
    {
        "scenario_id": "S4_known_noncontact_alen_manual_outbound",
        "alias": "KNOWN_NONCONTACT_1",
        "expected_direction": "outgoing_from_owner",
    },
    {
        "scenario_id": "S5_new_chat_inbound",
        "alias": "NEW_CHAT_1",
        "expected_direction": "incoming_to_owner",
    },
    {
        "scenario_id": "S6_new_chat_alen_manual_outbound",
        "alias": "NEW_CHAT_1",
        "expected_direction": "outgoing_from_owner",
    },
)

_MEETING_RE = re.compile(
    r"\b(встреча|созвон|звонок|колл|call|zoom|meet|meeting|appointment)\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"(\bзавтра\b|\bсегодня\b|\bпослезавтра\b|\bпонедельник\b|\bвторник\b|"
    r"\bсред[ау]\b|\bчетверг\b|\bпятниц[ау]\b|\bсуббот[ау]\b|\bвоскресенье\b|"
    r"\bmon(day)?\b|\btue(sday)?\b|\bwed(nesday)?\b|\bthu(rsday)?\b|"
    r"\bfri(day)?\b|\bsat(urday)?\b|\bsun(day)?\b|"
    r"\b\d{1,2}[:.]\d{2}\b|\bв\s+\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b)",
    re.IGNORECASE,
)
_DEADLINE_RE = re.compile(
    r"\b(deadline|due|дедлайн|до\s+\w+|надо\s+сдать|сдать\s+до|оплати|оплатить)\b",
    re.IGNORECASE,
)
_REMINDER_RE = re.compile(
    r"\b(напомни|не\s+забудь|не\s+забыть|remind|reminder)\b",
    re.IGNORECASE,
)
_FOLLOW_UP_RE = re.compile(
    r"\b(follow\s*up|ping|напиши|ответь|ответить|скинь|вернусь|позже|пингани)\b",
    re.IGNORECASE,
)
_RESERVATION_RE = re.compile(
    r"\b(бронь|брон[ьи]|reservation|ticket|билет|flight|hotel|booking)\b",
    re.IGNORECASE,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_list(values: Iterable[Any]) -> str:
    return _json_dumps([_coerce_text(value) for value in values])


def _direction_for_sender(sender_id: Any, owner_user_chat_id: Any) -> str:
    if sender_id is None or owner_user_chat_id is None:
        return "unknown"
    return "outgoing_from_owner" if str(sender_id) == str(owner_user_chat_id) else "incoming_to_owner"


def _coerce_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def detect_candidate_reasons(text: str | None) -> list[str]:
    """Return deterministic life-inbox candidate tags without persisting text.

    The tags are deliberately coarse. They let later jobs/extractors prioritize
    likely actionable messages while keeping raw private chat text out of the DB.
    """

    if not text:
        return []

    checks: list[tuple[str, re.Pattern[str]]] = [
        ("meeting", _MEETING_RE),
        ("time_reference", _TIME_RE),
        ("deadline", _DEADLINE_RE),
        ("reminder", _REMINDER_RE),
        ("follow_up", _FOLLOW_UP_RE),
        ("reservation", _RESERVATION_RE),
    ]
    return [name for name, pattern in checks if pattern.search(text)]


def default_life_home() -> Path:
    """Return the life-management root outside Hermes runtime state."""

    override = os.getenv("HERMES_LIFE_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".hermes-life"


def resolve_life_inbox_db_path(user_chat_id: str | int, *, life_home: Path | str | None = None) -> Path:
    """Resolve `life_inbox.sqlite` for a Telegram Business owner account.

    Raises KeyError/FileNotFoundError when the numeric Telegram user is not bound
    in `accounts.json`. Gateway call sites should catch and log this rather than
    falling back to a shared/global inbox.
    """

    root = Path(life_home).expanduser() if life_home is not None else default_life_home()
    registry_path = root / "accounts.json"
    data = json.loads(registry_path.read_text())
    key = f"telegram:{user_chat_id}"
    account = (data.get("accounts") or {}).get(key)
    if not account:
        raise KeyError(f"No life account registered for {key}")

    profile_rel = account.get("life_profile")
    if not profile_rel:
        raise KeyError(f"Life account {key} has no life_profile path")

    profile_path = Path(profile_rel)
    if not profile_path.is_absolute():
        profile_path = root / profile_path
    return profile_path.parent / "life_inbox.sqlite"


def _iter_account_db_paths(*, life_home: Path | str | None = None) -> Iterable[tuple[str, Path]]:
    root = Path(life_home).expanduser() if life_home is not None else default_life_home()
    data = json.loads((root / "accounts.json").read_text())
    for key, account in (data.get("accounts") or {}).items():
        if not str(key).startswith("telegram:"):
            continue
        profile_rel = account.get("life_profile") if isinstance(account, dict) else None
        if not profile_rel:
            continue
        profile_path = Path(profile_rel)
        if not profile_path.is_absolute():
            profile_path = root / profile_path
        yield key, profile_path.parent / "life_inbox.sqlite"


def resolve_business_connection_user_chat_id(
    connection_id: str | None,
    *,
    life_home: Path | str | None = None,
) -> str | None:
    """Find the owner user_chat_id for a previously stored Business connection.

    This lets gateway restarts continue storing business messages without using
    a process-global/shared inbox. The lookup only scans account-scoped DBs
    declared in `~/.hermes-life/accounts.json`.
    """

    if not connection_id:
        return None
    for _account_key, db_path in _iter_account_db_paths(life_home=life_home):
        if not db_path.exists():
            continue
        try:
            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    "SELECT user_chat_id FROM business_connections WHERE connection_id = ?",
                    (str(connection_id),),
                ).fetchone()
        except (OSError, sqlite3.DatabaseError):
            continue
        if row and row[0]:
            return str(row[0])
    return None


class LifeInboxStore:
    """Small SQLite store for selected personal/work message ingestion."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._secure_storage_permissions()
        self._init_db()
        self._secure_storage_permissions()

    def _secure_storage_permissions(self) -> None:
        """Best-effort private permissions for local life-inbox metadata."""
        if os.name == "nt":
            return
        try:
            self.db_path.parent.chmod(0o700)
        except OSError:
            pass
        for path in (self.db_path, Path(f"{self.db_path}-wal"), Path(f"{self.db_path}-shm")):
            try:
                if path.exists():
                    path.chmod(0o600)
            except OSError:
                pass

    @classmethod
    def for_telegram_user_chat_id(
        cls,
        user_chat_id: str | int,
        *,
        life_home: Path | str | None = None,
    ) -> "LifeInboxStore":
        return cls(resolve_life_inbox_db_path(user_chat_id, life_home=life_home))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode = WAL;

                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS source_chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    chat_name TEXT,
                    chat_type TEXT,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    UNIQUE(platform, chat_id)
                );

                CREATE TABLE IF NOT EXISTS chat_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    rule_mode TEXT NOT NULL DEFAULT 'metadata_only',
                    priority TEXT NOT NULL DEFAULT 'normal',
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(platform, chat_id)
                );

                CREATE TABLE IF NOT EXISTS business_connections (
                    connection_id TEXT PRIMARY KEY,
                    update_id INTEGER,
                    is_enabled INTEGER,
                    user_chat_id TEXT,
                    user_id TEXT,
                    username TEXT,
                    full_name TEXT,
                    rights_json TEXT,
                    first_seen_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS business_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform TEXT NOT NULL DEFAULT 'telegram_business',
                    update_id INTEGER,
                    update_type TEXT NOT NULL,
                    connection_id TEXT,
                    chat_id TEXT NOT NULL,
                    chat_type TEXT,
                    chat_name TEXT,
                    message_id TEXT NOT NULL,
                    sender_id TEXT,
                    sender_name TEXT,
                    message_date TEXT,
                    has_text INTEGER NOT NULL DEFAULT 0,
                    text_len INTEGER NOT NULL DEFAULT 0,
                    text_sha256 TEXT,
                    text_preview TEXT,
                    raw_text_stored INTEGER NOT NULL DEFAULT 0,
                    candidate_reasons_json TEXT NOT NULL DEFAULT '[]',
                    first_seen_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(connection_id, chat_id, message_id)
                );

                CREATE TABLE IF NOT EXISTS business_message_text (
                    business_message_id INTEGER PRIMARY KEY,
                    text TEXT NOT NULL,
                    text_sha256 TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(business_message_id) REFERENCES business_messages(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS business_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    username TEXT,
                    full_name TEXT,
                    is_bot INTEGER,
                    language_code TEXT,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    UNIQUE(platform, user_id)
                );

                CREATE TABLE IF NOT EXISTS chat_participants (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    UNIQUE(platform, chat_id, user_id)
                );

                CREATE INDEX IF NOT EXISTS idx_business_messages_chat_date
                    ON business_messages(chat_id, message_date);
                CREATE INDEX IF NOT EXISTS idx_business_messages_updated
                    ON business_messages(updated_at);
                CREATE INDEX IF NOT EXISTS idx_business_users_username
                    ON business_users(platform, username);
                CREATE INDEX IF NOT EXISTS idx_chat_participants_user
                    ON chat_participants(platform, user_id);

                CREATE TABLE IF NOT EXISTS business_payload_probe_scenarios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_lane TEXT NOT NULL,
                    scenario_id TEXT NOT NULL,
                    alias TEXT,
                    expected_direction TEXT,
                    probe_text_len INTEGER NOT NULL,
                    probe_text_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    matched_event_id INTEGER,
                    created_at TEXT NOT NULL,
                    matched_at TEXT,
                    notes TEXT,
                    UNIQUE(source_lane, scenario_id),
                    UNIQUE(source_lane, probe_text_sha256, probe_text_len)
                );

                CREATE INDEX IF NOT EXISTS idx_business_payload_probe_scenarios_status
                    ON business_payload_probe_scenarios(source_lane, status);

                CREATE TABLE IF NOT EXISTS business_payload_probe_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_lane TEXT NOT NULL,
                    scenario_id TEXT,
                    update_id INTEGER,
                    update_type TEXT NOT NULL,
                    connection_id TEXT,
                    chat_id TEXT,
                    message_id TEXT,
                    sender_id TEXT,
                    direction TEXT NOT NULL DEFAULT 'unknown',
                    message_date TEXT,
                    has_text INTEGER NOT NULL DEFAULT 0,
                    text_len INTEGER NOT NULL DEFAULT 0,
                    text_sha256 TEXT,
                    raw_text_stored INTEGER NOT NULL DEFAULT 0,
                    field_availability_json TEXT NOT NULL DEFAULT '{}',
                    payload_shape_json TEXT NOT NULL DEFAULT '{}',
                    media_json TEXT NOT NULL DEFAULT '{}',
                    reply_context_json TEXT NOT NULL DEFAULT '{}',
                    deleted_message_ids_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_business_payload_probe_events_scenario
                    ON business_payload_probe_events(source_lane, scenario_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_business_payload_probe_events_message
                    ON business_payload_probe_events(source_lane, connection_id, chat_id, message_id);

                CREATE TABLE IF NOT EXISTS deleted_business_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    update_id INTEGER,
                    connection_id TEXT,
                    chat_id TEXT,
                    message_ids_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS life_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER,
                    signal_type TEXT NOT NULL,
                    title TEXT,
                    confidence REAL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    extracted_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(message_id) REFERENCES business_messages(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    source TEXT,
                    payload_hash TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )

    def _ensure_chat(
        self,
        conn: sqlite3.Connection,
        *,
        platform: str,
        chat_id: str,
        chat_name: str | None,
        chat_type: str | None,
        now: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO source_chats(platform, chat_id, chat_name, chat_type, first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(platform, chat_id) DO UPDATE SET
                chat_name = excluded.chat_name,
                chat_type = excluded.chat_type,
                last_seen_at = excluded.last_seen_at
            """,
            (platform, chat_id, chat_name, chat_type, now, now),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO chat_rules(platform, chat_id, rule_mode, priority, created_at, updated_at)
            VALUES (?, ?, ?, 'normal', ?, ?)
            """,
            (platform, chat_id, DEFAULT_CHAT_RULE_MODE, now, now),
        )

    def _upsert_business_user(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: str | None,
        username: str | None,
        full_name: str | None,
        is_bot: bool | None,
        language_code: str | None,
        now: str,
    ) -> None:
        if not user_id:
            return
        conn.execute(
            """
            INSERT INTO business_users(
                platform, user_id, username, full_name, is_bot, language_code,
                first_seen_at, last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(platform, user_id) DO UPDATE SET
                username = COALESCE(excluded.username, business_users.username),
                full_name = COALESCE(excluded.full_name, business_users.full_name),
                is_bot = COALESCE(excluded.is_bot, business_users.is_bot),
                language_code = COALESCE(excluded.language_code, business_users.language_code),
                last_seen_at = excluded.last_seen_at
            """,
            (
                PLATFORM_TELEGRAM_BUSINESS,
                user_id,
                username,
                full_name,
                None if is_bot is None else int(bool(is_bot)),
                language_code,
                now,
                now,
            ),
        )

    def _ensure_chat_participant(
        self,
        conn: sqlite3.Connection,
        *,
        chat_id: str,
        user_id: str | None,
        now: str,
    ) -> None:
        if not user_id:
            return
        conn.execute(
            """
            INSERT INTO chat_participants(platform, chat_id, user_id, first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(platform, chat_id, user_id) DO UPDATE SET
                last_seen_at = excluded.last_seen_at
            """,
            (PLATFORM_TELEGRAM_BUSINESS, chat_id, user_id, now, now),
        )

    def prepare_business_payload_probe_scenarios(
        self,
        scenarios: Iterable[dict[str, Any]],
        *,
        source_lane: str = BUSINESS_PAYLOAD_PROBE_LANE,
    ) -> None:
        """Register strict live-probe scenario codes without storing raw text.

        `probe_text` is accepted only long enough to calculate SHA-256 + length.
        The plaintext code is intentionally not persisted, so later gateway
        ingestion can match probe messages by hash while keeping DB/logs free of
        raw chat content.
        """

        now = _utc_now_iso()
        with self._connect() as conn:
            for scenario in scenarios:
                scenario_id = str(scenario["scenario_id"])
                probe_text = str(scenario["probe_text"])
                conn.execute(
                    """
                    INSERT INTO business_payload_probe_scenarios(
                        source_lane, scenario_id, alias, expected_direction,
                        probe_text_len, probe_text_sha256, status,
                        matched_event_id, created_at, matched_at, notes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL, ?, NULL, ?)
                    ON CONFLICT(source_lane, scenario_id) DO UPDATE SET
                        alias = excluded.alias,
                        expected_direction = excluded.expected_direction,
                        probe_text_len = excluded.probe_text_len,
                        probe_text_sha256 = excluded.probe_text_sha256,
                        status = 'pending',
                        matched_event_id = NULL,
                        matched_at = NULL,
                        notes = excluded.notes
                    """,
                    (
                        source_lane,
                        scenario_id,
                        _coerce_text(scenario.get("alias")),
                        _coerce_text(scenario.get("expected_direction")),
                        len(probe_text),
                        _sha256_text(probe_text),
                        now,
                        _coerce_text(scenario.get("notes")),
                    ),
                )

    def _match_probe_scenario_by_text(
        self,
        conn: sqlite3.Connection,
        *,
        source_lane: str,
        text_sha256: str | None,
        text_len: int,
    ) -> sqlite3.Row | None:
        if not text_sha256:
            return None
        return conn.execute(
            """
            SELECT id, scenario_id
            FROM business_payload_probe_scenarios
            WHERE source_lane = ? AND probe_text_sha256 = ? AND probe_text_len = ?
            ORDER BY CASE status WHEN 'pending' THEN 0 WHEN 'matched' THEN 1 ELSE 2 END, id
            LIMIT 1
            """,
            (source_lane, text_sha256, text_len),
        ).fetchone()

    def _match_probe_scenario_by_message_identity(
        self,
        conn: sqlite3.Connection,
        *,
        source_lane: str,
        connection_id: str | None,
        chat_id: str | None,
        message_id: str | None,
        deleted_message_ids: Iterable[Any] | None = None,
    ) -> str | None:
        candidate_ids = []
        if message_id:
            candidate_ids.append(str(message_id))
        if deleted_message_ids:
            candidate_ids.extend(str(value) for value in deleted_message_ids if value is not None)
        if not candidate_ids:
            return None
        placeholders = ",".join("?" for _ in candidate_ids)
        params: list[Any] = [source_lane, connection_id, chat_id, *candidate_ids]
        row = conn.execute(
            f"""
            SELECT scenario_id
            FROM business_payload_probe_events
            WHERE source_lane = ?
              AND connection_id IS ?
              AND chat_id IS ?
              AND message_id IN ({placeholders})
              AND scenario_id IS NOT NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
        return str(row["scenario_id"]) if row and row["scenario_id"] else None

    def record_business_payload_probe_event(
        self,
        *,
        update_id: int | None,
        update_type: str,
        connection_id: str | None,
        owner_user_chat_id: str | int | None,
        chat_id: str | int | None,
        message_id: str | int | None = None,
        sender_id: str | int | None = None,
        text: str | None = None,
        message_date: Any = None,
        field_availability: Any | None = None,
        payload_shape: Any | None = None,
        media: Any | None = None,
        reply_context: Any | None = None,
        deleted_message_ids: Iterable[Any] | None = None,
        source_lane: str = BUSINESS_PAYLOAD_PROBE_LANE,
        capture_all: bool = False,
    ) -> int | None:
        """Record a sanitized Telegram Business payload-probe event.

        By default the store records only events that match a prepared scenario
        code by SHA-256/length, or follow-up edit/delete events for an already
        matched message. Set `capture_all=True` for temporary shape-only capture
        of all Business payloads. Raw text is never stored.
        """

        now = _utc_now_iso()
        text_value = text or ""
        text_len = len(text_value)
        text_sha256 = _sha256_text(text_value) if text_value else None
        connection_key = _coerce_text(connection_id)
        chat_id_text = _coerce_text(chat_id)
        message_id_text = _coerce_text(message_id)
        sender_id_text = _coerce_text(sender_id)
        deleted_ids = [value for value in (deleted_message_ids or [])]

        with self._connect() as conn:
            scenario_row = self._match_probe_scenario_by_text(
                conn,
                source_lane=source_lane,
                text_sha256=text_sha256,
                text_len=text_len,
            )
            scenario_id = str(scenario_row["scenario_id"]) if scenario_row else None
            if scenario_id is None:
                scenario_id = self._match_probe_scenario_by_message_identity(
                    conn,
                    source_lane=source_lane,
                    connection_id=connection_key,
                    chat_id=chat_id_text,
                    message_id=message_id_text,
                    deleted_message_ids=deleted_ids,
                )
            if scenario_id is None and not capture_all:
                return None

            cur = conn.execute(
                """
                INSERT INTO business_payload_probe_events(
                    source_lane, scenario_id, update_id, update_type, connection_id,
                    chat_id, message_id, sender_id, direction, message_date,
                    has_text, text_len, text_sha256, raw_text_stored,
                    field_availability_json, payload_shape_json, media_json,
                    reply_context_json, deleted_message_ids_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_lane,
                    scenario_id,
                    update_id,
                    update_type,
                    connection_key,
                    chat_id_text,
                    message_id_text,
                    sender_id_text,
                    _direction_for_sender(sender_id_text, owner_user_chat_id),
                    _coerce_iso(message_date),
                    int(bool(text_value)),
                    text_len,
                    text_sha256,
                    _json_dumps(field_availability or {}),
                    _json_dumps(payload_shape or {}),
                    _json_dumps(media or {}),
                    _json_dumps(reply_context or {}),
                    _json_list(deleted_ids),
                    now,
                ),
            )
            if cur.lastrowid is None:
                raise RuntimeError("payload probe event insert succeeded but row id is unavailable")
            event_id = int(cur.lastrowid)
            if scenario_row is not None:
                conn.execute(
                    """
                    UPDATE business_payload_probe_scenarios
                    SET status = 'matched', matched_event_id = ?, matched_at = ?
                    WHERE id = ?
                    """,
                    (event_id, now, int(scenario_row["id"])),
                )
            elif scenario_id is not None:
                conn.execute(
                    """
                    UPDATE business_payload_probe_scenarios
                    SET status = 'matched', matched_event_id = COALESCE(matched_event_id, ?),
                        matched_at = COALESCE(matched_at, ?)
                    WHERE source_lane = ? AND scenario_id = ?
                    """,
                    (event_id, now, source_lane, scenario_id),
                )
            return event_id

    def record_business_connection(
        self,
        *,
        update_id: int | None,
        connection_id: str,
        is_enabled: bool | None,
        user_chat_id: str | int | None,
        user_id: str | int | None,
        username: str | None,
        full_name: str | None,
        rights: Any,
    ) -> None:
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO business_connections(
                    connection_id, update_id, is_enabled, user_chat_id, user_id,
                    username, full_name, rights_json, first_seen_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(connection_id) DO UPDATE SET
                    update_id = excluded.update_id,
                    is_enabled = excluded.is_enabled,
                    user_chat_id = excluded.user_chat_id,
                    user_id = excluded.user_id,
                    username = excluded.username,
                    full_name = excluded.full_name,
                    rights_json = excluded.rights_json,
                    updated_at = excluded.updated_at
                """,
                (
                    connection_id,
                    update_id,
                    None if is_enabled is None else int(bool(is_enabled)),
                    _coerce_text(user_chat_id),
                    _coerce_text(user_id),
                    username,
                    full_name,
                    _json_dumps(rights),
                    now,
                    now,
                ),
            )

    def record_business_message(
        self,
        *,
        update_id: int | None,
        update_type: str,
        connection_id: str | None,
        chat_id: str | int | None,
        chat_type: str | None,
        chat_name: str | None,
        chat_username: str | None = None,
        message_id: str | int | None,
        sender_id: str | int | None,
        sender_name: str | None,
        sender_username: str | None = None,
        sender_is_bot: bool | None = None,
        sender_language_code: str | None = None,
        text: str | None,
        message_date: Any,
    ) -> int:
        now = _utc_now_iso()
        connection_key = _coerce_text(connection_id)
        chat_id_text = _coerce_text(chat_id)
        message_id_text = _coerce_text(message_id)
        sender_id_text = _coerce_text(sender_id)
        chat_username_text = _coerce_text(chat_username)
        sender_username_text = _coerce_text(sender_username)
        sender_language_code_text = _coerce_text(sender_language_code)
        if not connection_key:
            raise ValueError("connection_id is required for Telegram Business messages")
        if not chat_id_text:
            raise ValueError("chat_id is required for Telegram Business messages")
        if not message_id_text:
            raise ValueError("message_id is required for Telegram Business messages")
        text_value = text or ""
        text_sha256 = _sha256_text(text_value) if text_value else None
        candidate_reasons = detect_candidate_reasons(text_value)

        with self._connect() as conn:
            self._ensure_chat(
                conn,
                platform=PLATFORM_TELEGRAM_BUSINESS,
                chat_id=chat_id_text,
                chat_name=chat_name,
                chat_type=chat_type,
                now=now,
            )
            if str(chat_type or "").lower() == "private":
                self._upsert_business_user(
                    conn,
                    user_id=chat_id_text,
                    username=chat_username_text,
                    full_name=chat_name,
                    is_bot=None,
                    language_code=None,
                    now=now,
                )
                self._ensure_chat_participant(conn, chat_id=chat_id_text, user_id=chat_id_text, now=now)
            self._upsert_business_user(
                conn,
                user_id=sender_id_text,
                username=sender_username_text,
                full_name=sender_name,
                is_bot=sender_is_bot,
                language_code=sender_language_code_text,
                now=now,
            )
            self._ensure_chat_participant(conn, chat_id=chat_id_text, user_id=sender_id_text, now=now)
            conn.execute(
                """
                INSERT INTO business_messages(
                    platform, update_id, update_type, connection_id, chat_id, chat_type, chat_name,
                    message_id, sender_id, sender_name, message_date, has_text, text_len,
                    text_sha256, text_preview, raw_text_stored, candidate_reasons_json,
                    first_seen_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
                ON CONFLICT(connection_id, chat_id, message_id) DO UPDATE SET
                    update_id = excluded.update_id,
                    update_type = excluded.update_type,
                    chat_type = excluded.chat_type,
                    chat_name = excluded.chat_name,
                    sender_id = excluded.sender_id,
                    sender_name = excluded.sender_name,
                    message_date = excluded.message_date,
                    has_text = excluded.has_text,
                    text_len = excluded.text_len,
                    text_sha256 = excluded.text_sha256,
                    text_preview = NULL,
                    raw_text_stored = excluded.raw_text_stored,
                    candidate_reasons_json = excluded.candidate_reasons_json,
                    updated_at = excluded.updated_at
                """,
                (
                    PLATFORM_TELEGRAM_BUSINESS,
                    update_id,
                    update_type,
                    connection_key,
                    chat_id_text,
                    chat_type,
                    chat_name,
                    message_id_text,
                    sender_id_text,
                    sender_name,
                    _coerce_iso(message_date),
                    int(bool(text_value)),
                    len(text_value),
                    text_sha256,
                    int(bool(text_value)),
                    _json_dumps(candidate_reasons),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                """
                SELECT id FROM business_messages
                WHERE connection_id IS ? AND chat_id = ? AND message_id = ?
                """,
                (connection_key, chat_id_text, message_id_text),
            ).fetchone()
            if row is None:
                raise RuntimeError("business message insert succeeded but row lookup failed")
            business_message_id = int(row["id"])
            if text_value:
                conn.execute(
                    """
                    INSERT INTO business_message_text(
                        business_message_id, text, text_sha256, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(business_message_id) DO UPDATE SET
                        text = excluded.text,
                        text_sha256 = excluded.text_sha256,
                        updated_at = excluded.updated_at
                    """,
                    (business_message_id, text_value, text_sha256, now, now),
                )
            else:
                conn.execute(
                    "DELETE FROM business_message_text WHERE business_message_id = ?",
                    (business_message_id,),
                )
            return business_message_id

    def record_deleted_business_messages(
        self,
        *,
        update_id: int | None,
        connection_id: str | None,
        chat_id: str | int | None,
        message_ids: Iterable[Any],
    ) -> None:
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO deleted_business_messages(update_id, connection_id, chat_id, message_ids_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    update_id,
                    connection_id,
                    _coerce_text(chat_id),
                    _json_dumps([_coerce_text(message_id) for message_id in message_ids]),
                    now,
                ),
            )

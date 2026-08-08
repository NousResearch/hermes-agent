"""Platform-neutral interactive actions for external Hermes plugins.

This module deliberately sits at the gateway/plugin edge.  Interactive
actions are operator control-plane events, not model messages or tools.
"""

from __future__ import annotations

import re
import json
import asyncio
import inspect
import logging
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Literal, Mapping, cast

from hermes_constants import get_hermes_home


AuthorizationPolicy = Literal["initiator_only", "authorized_user"]

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SECRET_FIELD_RE = re.compile(
    r"(?:^|_)(?:api_?key|authorization|cookie|credential|password|secret|token)(?:$|_)",
    re.IGNORECASE,
)


class InteractiveActionRegistrationError(ValueError):
    """The plugin supplied an invalid or conflicting action registration."""


class InteractiveCardValidationError(ValueError):
    """A platform-neutral card envelope violates the bounded public contract."""


class InteractiveCardUnavailableError(RuntimeError):
    """No current gateway turn is bound to the plugin capability."""


class InteractiveCardDeliveryError(RuntimeError):
    """A sanitized card or fallback delivery failure."""


class InteractiveActionCapacityError(RuntimeError):
    """The bounded profile ledger has no safe room for another action."""


class InteractiveActionConflictError(Exception):
    """A handler-declared terminal conflict with safe operator-facing text."""

    def __init__(self, public_message: str = "") -> None:
        super().__init__("interactive action conflict")
        self.public_message = _sanitize_public_message(public_message)


class InteractiveActionRetryableError(Exception):
    """A handler-declared transient failure with safe operator-facing text."""

    def __init__(self, public_message: str = "") -> None:
        super().__init__("interactive action retryable failure")
        self.public_message = _sanitize_public_message(public_message)


def _sanitize_public_message(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:240]


def _bounded_text(
    value: object,
    *,
    field: str,
    limit: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise InteractiveCardValidationError(f"{field} must be text")
    if not allow_empty and not value.strip():
        raise InteractiveCardValidationError(f"{field} must not be empty")
    if len(value) > limit:
        raise InteractiveCardValidationError(f"{field} exceeds {limit} characters")
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({
            key: _freeze_json(item) for key, item in value.items()
        })
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _reject_secret_fields(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise InteractiveCardValidationError(
                    "action payload object keys must be text"
                )
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if _SECRET_FIELD_RE.search(normalized):
                raise InteractiveCardValidationError(
                    "action payload must not contain secret fields"
                )
            _reject_secret_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_secret_fields(item)


def _immutable_json_object(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InteractiveCardValidationError("action payload must be a JSON object")
    _reject_secret_fields(value)
    plain = _thaw_json(value)
    try:
        encoded = json.dumps(
            plain,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InteractiveCardValidationError(
            "action payload must contain only JSON values"
        ) from exc
    if len(encoded) > 16_384:
        raise InteractiveCardValidationError("action payload exceeds 16384 bytes")
    frozen = _freeze_json(plain)
    assert isinstance(frozen, Mapping)
    return cast(Mapping[str, object], frozen)


@dataclass(frozen=True)
class InteractiveCardFact:
    """One short label/value pair rendered by a native adapter."""

    label: str
    value: str

    def __post_init__(self) -> None:
        _bounded_text(self.label, field="fact label", limit=80)
        _bounded_text(self.value, field="fact value", limit=300)


@dataclass(frozen=True)
class InteractiveCardSection:
    """One bounded explanatory section."""

    title: str
    body: str

    def __post_init__(self) -> None:
        _bounded_text(self.title, field="section title", limit=80)
        _bounded_text(self.body, field="section body", limit=1_500)


@dataclass(frozen=True)
class InteractiveCardAction:
    """A plugin action whose payload remains server-side."""

    label: str
    action: str
    external_action_id: str
    payload: Mapping[str, object]
    style: Literal["default", "primary", "danger"] = "default"

    def __post_init__(self) -> None:
        _bounded_text(self.label, field="action label", limit=80)
        _bounded_text(self.action, field="action", limit=320)
        if "/" not in self.action:
            raise InteractiveCardValidationError(
                "action must be a qualified plugin action"
            )
        plugin_id, local_name = self.action.rsplit("/", 1)
        try:
            canonical_action = qualify_action_name(plugin_id, local_name)
        except InteractiveActionRegistrationError as exc:
            raise InteractiveCardValidationError(str(exc)) from exc
        object.__setattr__(self, "action", canonical_action)
        _bounded_text(
            self.external_action_id,
            field="external action id",
            limit=128,
        )
        if self.style not in {"default", "primary", "danger"}:
            raise InteractiveCardValidationError("unsupported action style")
        object.__setattr__(self, "payload", _immutable_json_object(self.payload))


@dataclass(frozen=True)
class InteractiveCardEnvelope:
    """Versioned, bounded data contract shared by plugins and adapters."""

    version: int
    title: str
    summary: str
    fallback_text: str
    expires_in_seconds: int
    actions: tuple[InteractiveCardAction, ...]
    facts: tuple[InteractiveCardFact, ...] = ()
    sections: tuple[InteractiveCardSection, ...] = ()

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise InteractiveCardValidationError("unsupported interactive card version")
        _bounded_text(self.title, field="title", limit=120)
        _bounded_text(self.summary, field="summary", limit=1_000)
        _bounded_text(self.fallback_text, field="fallback text", limit=4_000)
        if type(self.expires_in_seconds) is not int or not (
            1 <= self.expires_in_seconds <= 86_400
        ):
            raise InteractiveCardValidationError(
                "expires_in_seconds must be between 1 and 86400"
            )
        object.__setattr__(self, "facts", tuple(self.facts))
        object.__setattr__(self, "sections", tuple(self.sections))
        object.__setattr__(self, "actions", tuple(self.actions))
        if len(self.facts) > 12:
            raise InteractiveCardValidationError("card has more than 12 facts")
        if len(self.sections) > 8:
            raise InteractiveCardValidationError("card has more than 8 sections")
        if not 1 <= len(self.actions) <= 5:
            raise InteractiveCardValidationError(
                "card must have between 1 and 5 actions"
            )
        if not all(isinstance(item, InteractiveCardFact) for item in self.facts):
            raise InteractiveCardValidationError(
                "facts must contain InteractiveCardFact values"
            )
        if not all(isinstance(item, InteractiveCardSection) for item in self.sections):
            raise InteractiveCardValidationError(
                "sections must contain InteractiveCardSection values"
            )
        if not all(isinstance(item, InteractiveCardAction) for item in self.actions):
            raise InteractiveCardValidationError(
                "actions must contain InteractiveCardAction values"
            )


@dataclass(frozen=True)
class InteractiveCardOrigin:
    """Gateway-owned origin bound to a plugin send capability."""

    platform: str
    profile_id: str
    chat_id: str
    thread_id: str | None
    initiator_id: str
    initiator_name: str
    message_id: str

    def __post_init__(self) -> None:
        _bounded_text(self.platform, field="origin platform", limit=64)
        _bounded_text(self.profile_id, field="origin profile", limit=128)
        _bounded_text(self.chat_id, field="origin chat", limit=512)
        _bounded_text(
            self.initiator_id,
            field="origin initiator",
            limit=512,
            allow_empty=True,
        )
        _bounded_text(
            self.initiator_name,
            field="origin initiator name",
            limit=256,
            allow_empty=True,
        )
        _bounded_text(
            self.message_id,
            field="origin message",
            limit=512,
            allow_empty=True,
        )
        if self.thread_id is not None:
            _bounded_text(self.thread_id, field="origin thread", limit=512)


@dataclass(frozen=True)
class InteractiveActionCallback:
    """Authenticated, platform-normalized input to action dispatch."""

    action_instance_id: str
    platform: str
    profile_id: str
    operator_id: str
    operator_name: str
    chat_id: str
    thread_id: str | None
    card_id: str


@dataclass(frozen=True)
class InteractiveActionContext:
    """Immutable context passed to an external plugin action handler."""

    platform: str
    profile_id: str
    operator_id: str
    operator_name: str
    chat_id: str
    thread_id: str | None
    message_id: str
    card_id: str
    action_instance_id: str
    external_action_id: str
    payload: Mapping[str, object]


InteractiveActionStatus = Literal[
    "processing",
    "succeeded",
    "downstream_replay",
    "already_processed",
    "denied",
    "unknown",
    "expired",
    "conflict",
    "retryable_failure",
    "unknown_outcome",
]


_DEFAULT_RESULT_MESSAGES: dict[str, str] = {
    "processing": "Processing confirmation…",
    "succeeded": "Applied successfully.",
    "downstream_replay": "Already applied downstream.",
    "already_processed": "Already processed.",
    "denied": "You are not authorized to apply this confirmation.",
    "unknown": "This confirmation is unknown or no longer available.",
    "expired": "This confirmation has expired.",
    "conflict": "The confirmation could not be applied because its state changed.",
    "retryable_failure": (
        "The confirmation could not be completed right now. "
        "Use the same confirmation to retry."
    ),
    "unknown_outcome": (
        "Hermes could not verify whether this confirmation took effect. "
        "Do not retry it; verify the downstream state first."
    ),
}


@dataclass(frozen=True)
class InteractiveActionResult:
    """Truthful, sanitized outcome for callback UX and durable replay state."""

    status: InteractiveActionStatus
    user_message: str

    def __post_init__(self) -> None:
        if self.status not in _DEFAULT_RESULT_MESSAGES:
            raise ValueError("invalid interactive action result status")
        object.__setattr__(
            self,
            "user_message",
            _sanitize_public_message(self.user_message)
            or _DEFAULT_RESULT_MESSAGES[self.status],
        )

    @classmethod
    def _make(
        cls, status: InteractiveActionStatus, message: str = ""
    ) -> "InteractiveActionResult":
        return cls(
            status=status,
            user_message=_sanitize_public_message(message)
            or _DEFAULT_RESULT_MESSAGES[status],
        )

    @classmethod
    def processing(cls, message: str = "") -> "InteractiveActionResult":
        return cls._make("processing", message)

    @classmethod
    def succeeded(cls, message: str = "") -> "InteractiveActionResult":
        return cls._make("succeeded", message)

    @classmethod
    def downstream_replay(cls, message: str = "") -> "InteractiveActionResult":
        return cls._make("downstream_replay", message)

    @classmethod
    def already_processed(cls) -> "InteractiveActionResult":
        return cls._make("already_processed")

    @classmethod
    def denied(cls) -> "InteractiveActionResult":
        return cls._make("denied")

    @classmethod
    def unknown(cls) -> "InteractiveActionResult":
        return cls._make("unknown")

    @classmethod
    def expired(cls) -> "InteractiveActionResult":
        return cls._make("expired")

    @classmethod
    def conflict(cls, message: str = "") -> "InteractiveActionResult":
        return cls._make("conflict", message)

    @classmethod
    def retryable_failure(cls, message: str = "") -> "InteractiveActionResult":
        return cls._make("retryable_failure", message)

    @classmethod
    def unknown_outcome(cls) -> "InteractiveActionResult":
        return cls._make("unknown_outcome")


@dataclass(frozen=True)
class PreparedInteractiveCard:
    """Opaque action IDs reserved before native card delivery."""

    action_instance_ids: tuple[str, ...]


@dataclass(frozen=True)
class InteractiveCardDelivery:
    """Bounded receipt returned to the external plugin."""

    mode: Literal["native", "fallback"]
    message_id: str
    action_instance_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _StoredAction:
    action_instance_id: str
    plugin_action: str
    external_action_id: str
    authorization_policy: AuthorizationPolicy
    profile_id: str
    platform: str
    chat_id: str
    thread_id: str | None
    initiator_id: str
    initiator_name: str
    message_id: str
    card_id: str | None
    payload_json: str
    state: str
    outcome: str | None
    user_message: str
    expires_at: float

    def handler_context(
        self, callback: InteractiveActionCallback
    ) -> InteractiveActionContext:
        payload = json.loads(self.payload_json)
        return InteractiveActionContext(
            platform=self.platform,
            profile_id=self.profile_id,
            operator_id=callback.operator_id,
            operator_name=callback.operator_name,
            chat_id=self.chat_id,
            thread_id=self.thread_id,
            message_id=self.message_id,
            card_id=self.card_id or "",
            action_instance_id=self.action_instance_id,
            external_action_id=self.external_action_id,
            payload=_immutable_json_object(payload),
        )


@dataclass(frozen=True)
class _ClaimResult:
    status: Literal[
        "claimed",
        "processing",
        "finished",
        "denied",
        "unknown",
        "expired",
    ]
    record: _StoredAction | None = None


@dataclass(frozen=True)
class ClaimedInteractiveAction:
    """Opaque manager-owned claim passed from callback ACK to completion."""

    callback: InteractiveActionCallback
    record: _StoredAction
    registration: InteractiveActionRegistration


class SQLiteInteractiveActionStorage:
    """Bounded profile-safe SQLite action ledger with atomic claims."""

    _RETENTION_SECONDS = 7 * 24 * 60 * 60
    _MAX_ROWS = 1_000
    # Feishu's synchronous card callback waits at most three seconds for
    # authorization + claim.  Database contention must fail inside that
    # window so the adapter can return a truthful retryable result instead of
    # timing out while a claim later succeeds invisibly.
    _SQLITE_TIMEOUT_SECONDS = 0.5
    _REQUIRED_COLUMNS = frozenset({
        "action_instance_id",
        "plugin_action",
        "external_action_id",
        "authorization_policy",
        "profile_id",
        "platform",
        "chat_id",
        "thread_id",
        "initiator_id",
        "initiator_name",
        "message_id",
        "card_id",
        "payload_json",
        "state",
        "outcome",
        "user_message",
        "expires_at",
        "created_at",
        "updated_at",
    })

    def __init__(self, db_path: Path | str | Callable[[], Path] | None = None) -> None:
        self._db_path_provider: Callable[[], Path]
        if db_path is None:
            self._db_path_provider = lambda: get_hermes_home() / "state.db"
        elif callable(db_path):
            self._db_path_provider = cast(Callable[[], Path], db_path)
        else:
            resolved_path = Path(db_path)
            self._db_path_provider = lambda: resolved_path
        self._lock = threading.Lock()
        self._schema_ready_paths: set[str] = set()

    def _path(self) -> Path:
        return self._db_path_provider()

    def _connect(self) -> sqlite3.Connection:
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            path,
            timeout=self._SQLITE_TIMEOUT_SECONDS,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        try:
            from hermes_state import apply_wal_with_fallback

            apply_wal_with_fallback(conn, db_label="state.db (interactive_actions)")
            path_key = str(path.resolve(strict=False))
            if path_key not in self._schema_ready_paths:
                self._ensure_schema(conn)
                self._reconcile_processing_from_prior_process(conn)
                self._schema_ready_paths.add(path_key)
        except Exception:
            conn.close()
            raise
        return conn

    @classmethod
    def _schema_columns(cls, conn: sqlite3.Connection) -> set[str]:
        return {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(interactive_actions)")
        }

    @classmethod
    def _schema_is_current(cls, conn: sqlite3.Connection) -> bool:
        columns = cls._schema_columns(conn)
        if not cls._REQUIRED_COLUMNS.issubset(columns):
            return False
        indexes = {
            str(row[1])
            for row in conn.execute("PRAGMA index_list(interactive_actions)")
        }
        return "idx_interactive_actions_expiry" in indexes

    @classmethod
    def _ensure_schema(cls, conn: sqlite3.Connection) -> None:
        """Create or migrate the ledger without racing another process.

        The read-only fast path avoids taking a schema write lock on every
        callback after restart.  The transaction then re-checks all schema
        facts after ``BEGIN IMMEDIATE`` so concurrent old-schema migrations
        cannot both attempt the same ``ALTER TABLE``.
        """

        if cls._schema_is_current(conn):
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS interactive_actions (
                    action_instance_id TEXT PRIMARY KEY,
                    plugin_action TEXT NOT NULL,
                    external_action_id TEXT NOT NULL,
                    authorization_policy TEXT NOT NULL DEFAULT 'initiator_only',
                    profile_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    thread_id TEXT,
                    initiator_id TEXT NOT NULL,
                    initiator_name TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    card_id TEXT,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    outcome TEXT,
                    user_message TEXT NOT NULL DEFAULT '',
                    expires_at REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )"""
            )
            columns = cls._schema_columns(conn)
            if "authorization_policy" not in columns:
                conn.execute(
                    """ALTER TABLE interactive_actions
                       ADD COLUMN authorization_policy TEXT NOT NULL
                       DEFAULT 'initiator_only'"""
                )
                columns.add("authorization_policy")
            if "user_message" not in columns:
                conn.execute(
                    """ALTER TABLE interactive_actions
                       ADD COLUMN user_message TEXT NOT NULL DEFAULT ''"""
                )
                columns.add("user_message")
            missing = cls._REQUIRED_COLUMNS - columns
            if missing:
                missing_names = ", ".join(sorted(missing))
                raise sqlite3.DatabaseError(
                    f"interactive_actions schema is missing columns: {missing_names}"
                )
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_interactive_actions_expiry
                   ON interactive_actions(expires_at, state)"""
            )
            for status, message in _DEFAULT_RESULT_MESSAGES.items():
                conn.execute(
                    """UPDATE interactive_actions
                       SET user_message=?
                       WHERE state='finished' AND outcome=? AND user_message=''""",
                    (message, status),
                )
            conn.execute(
                """UPDATE interactive_actions
                   SET outcome='already_processed', user_message=?
                   WHERE state='finished' AND outcome IS NULL""",
                (_DEFAULT_RESULT_MESSAGES["already_processed"],),
            )
            conn.execute(
                """UPDATE interactive_actions
                   SET state='active'
                   WHERE state='finished' AND outcome='retryable_failure'"""
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @classmethod
    def _reconcile_processing_from_prior_process(
        cls,
        conn: sqlite3.Connection,
    ) -> None:
        """Terminalize crash-left processing rows once for this storage startup."""

        if conn.execute(
            "SELECT 1 FROM interactive_actions WHERE state='processing' LIMIT 1"
        ).fetchone() is None:
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                """UPDATE interactive_actions
                   SET state='finished', outcome='unknown_outcome',
                       user_message=?, updated_at=?
                   WHERE state='processing'""",
                (_DEFAULT_RESULT_MESSAGES["unknown_outcome"], time.time()),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @staticmethod
    def _row(row: sqlite3.Row | None) -> _StoredAction | None:
        if row is None:
            return None
        return _StoredAction(
            action_instance_id=row["action_instance_id"],
            plugin_action=row["plugin_action"],
            external_action_id=row["external_action_id"],
            authorization_policy=row["authorization_policy"],
            profile_id=row["profile_id"],
            platform=row["platform"],
            chat_id=row["chat_id"],
            thread_id=row["thread_id"],
            initiator_id=row["initiator_id"],
            initiator_name=row["initiator_name"],
            message_id=row["message_id"],
            card_id=row["card_id"],
            payload_json=row["payload_json"],
            state=row["state"],
            outcome=row["outcome"],
            user_message=row["user_message"],
            expires_at=float(row["expires_at"]),
        )

    def reserve(
        self,
        rows: list[
            tuple[
                str,
                InteractiveCardAction,
                InteractiveActionRegistration,
            ]
        ],
        *,
        origin: InteractiveCardOrigin,
        expires_at: float,
        now: float,
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                self._prune(conn, now)
                count = int(
                    conn.execute("SELECT COUNT(*) FROM interactive_actions").fetchone()[
                        0
                    ]
                )
                if count + len(rows) > self._MAX_ROWS:
                    raise InteractiveActionCapacityError(
                        "interactive action ledger reached its bounded capacity"
                    )
                for instance_id, action, registration in rows:
                    payload_json = json.dumps(
                        _thaw_json(action.payload),
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    conn.execute(
                        """INSERT INTO interactive_actions (
                            action_instance_id, plugin_action, external_action_id,
                            authorization_policy,
                            profile_id, platform, chat_id, thread_id,
                            initiator_id, initiator_name, message_id, card_id,
                            payload_json, state, outcome, expires_at,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?,
                                  'awaiting_delivery', NULL, ?, ?, ?)""",
                        (
                            instance_id,
                            action.action,
                            action.external_action_id,
                            registration.authorization_policy,
                            origin.profile_id,
                            origin.platform,
                            origin.chat_id,
                            origin.thread_id,
                            origin.initiator_id,
                            origin.initiator_name,
                            origin.message_id,
                            payload_json,
                            expires_at,
                            now,
                            now,
                        ),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _prune(self, conn: sqlite3.Connection, now: float) -> None:
        cutoff = now - self._RETENTION_SECONDS
        conn.execute(
            """DELETE FROM interactive_actions
               WHERE updated_at < ? AND state IN ('finished', 'expired', 'delivery_failed')""",
            (cutoff,),
        )
        # A crash between reserve/send/activate leaves an awaiting-delivery
        # row; a crash after claim leaves a processing row that must never be
        # executed again.  Retain both long enough for audit/replay safety,
        # then discard them so repeated crashes cannot permanently exhaust the
        # bounded ledger.
        conn.execute(
            """DELETE FROM interactive_actions
               WHERE (state='processing' AND updated_at < ?)
                  OR (state IN ('active', 'awaiting_delivery') AND expires_at < ?)""",
            (cutoff, cutoff),
        )
        conn.execute(
            """UPDATE interactive_actions
               SET state='expired', updated_at=?
               WHERE state IN ('active', 'awaiting_delivery') AND expires_at <= ?""",
            (now, now),
        )
        count = int(
            conn.execute("SELECT COUNT(*) FROM interactive_actions").fetchone()[0]
        )
        overflow = count - self._MAX_ROWS
        if overflow > 0:
            conn.execute(
                """DELETE FROM interactive_actions WHERE action_instance_id IN (
                       SELECT action_instance_id FROM interactive_actions
                       WHERE state IN ('finished', 'expired', 'delivery_failed')
                       ORDER BY updated_at ASC LIMIT ?
                   )""",
                (overflow,),
            )

    def activate(
        self, action_instance_ids: tuple[str, ...], *, card_id: str, now: float
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                for instance_id in action_instance_ids:
                    cursor = conn.execute(
                        """UPDATE interactive_actions
                           SET state='active', card_id=?, updated_at=?
                           WHERE action_instance_id=? AND state='awaiting_delivery'""",
                        (card_id, now, instance_id),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("interactive action could not be activated")
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def fail_delivery(
        self, action_instance_ids: tuple[str, ...], *, now: float
    ) -> None:
        if not action_instance_ids:
            return
        placeholders = ",".join("?" for _ in action_instance_ids)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    f"""UPDATE interactive_actions
                        SET state='delivery_failed', updated_at=?
                        WHERE state='awaiting_delivery'
                          AND action_instance_id IN ({placeholders})""",  # noqa: S608 - placeholders only
                    (now, *action_instance_ids),
                )
                conn.commit()
            finally:
                conn.close()

    def get(self, action_instance_id: str) -> _StoredAction | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM interactive_actions WHERE action_instance_id=?",
                    (action_instance_id,),
                ).fetchone()
                return self._row(row)
            finally:
                conn.close()

    def claim(self, callback: InteractiveActionCallback, *, now: float) -> _ClaimResult:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM interactive_actions WHERE action_instance_id=?",
                    (callback.action_instance_id,),
                ).fetchone()
                record = self._row(row)
                if record is None:
                    conn.rollback()
                    return _ClaimResult("unknown")
                binding_matches = (
                    callback.profile_id == record.profile_id
                    and callback.platform == record.platform
                    and callback.chat_id == record.chat_id
                    # Feishu callback payloads do not always expose the topic
                    # root. When absent, the verified outbound card ID is the
                    # thread anchor; when present, require an exact match.
                    and (
                        callback.thread_id is None
                        or callback.thread_id == record.thread_id
                    )
                    and callback.card_id == record.card_id
                )
                if not binding_matches:
                    conn.rollback()
                    return _ClaimResult("denied", record)
                if record.state == "processing":
                    conn.rollback()
                    return _ClaimResult("processing", record)
                if record.state == "finished":
                    conn.rollback()
                    return _ClaimResult("finished", record)
                if now >= record.expires_at:
                    conn.execute(
                        """UPDATE interactive_actions
                           SET state='expired', updated_at=?
                           WHERE action_instance_id=? AND state IN ('active', 'awaiting_delivery')""",
                        (now, callback.action_instance_id),
                    )
                    conn.commit()
                    return _ClaimResult("expired", record)
                if record.state != "active":
                    conn.rollback()
                    return _ClaimResult("denied", record)
                cursor = conn.execute(
                    """UPDATE interactive_actions
                       SET state='processing', updated_at=?
                       WHERE action_instance_id=? AND state='active'""",
                    (now, callback.action_instance_id),
                )
                if cursor.rowcount != 1:
                    conn.rollback()
                    return _ClaimResult("processing", record)
                conn.commit()
                return _ClaimResult("claimed", record)
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def finish(
        self,
        action_instance_id: str,
        *,
        result: InteractiveActionResult,
        now: float,
    ) -> bool:
        next_state = "active" if result.status == "retryable_failure" else "finished"
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """UPDATE interactive_actions
                       SET state=?, outcome=?, user_message=?, updated_at=?
                       WHERE action_instance_id=? AND state='processing'""",
                    (
                        next_state,
                        result.status,
                        _sanitize_public_message(result.user_message),
                        now,
                        action_instance_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount == 1
            finally:
                conn.close()


class InteractiveActionManager:
    """Small orchestration seam for plugin registration, ledger and execution."""

    def __init__(
        self,
        *,
        storage: SQLiteInteractiveActionStorage | None = None,
        registrations: Mapping[str, InteractiveActionRegistration]
        | Callable[[], Mapping[str, InteractiveActionRegistration]],
        clock: Callable[[], float] = time.time,
        id_source: Callable[[], str] = lambda: secrets.token_urlsafe(24),
    ) -> None:
        self.storage = storage or SQLiteInteractiveActionStorage()
        self._registrations_provider: Callable[
            [], Mapping[str, InteractiveActionRegistration]
        ]
        if callable(registrations):
            self._registrations_provider = cast(
                Callable[[], Mapping[str, InteractiveActionRegistration]],
                registrations,
            )
        else:
            static_registrations = cast(
                Mapping[str, InteractiveActionRegistration],
                registrations,
            )
            self._registrations_provider = lambda: static_registrations
        self._clock = clock
        self._id_source = id_source

    def _registry(self) -> Mapping[str, InteractiveActionRegistration]:
        return self._registrations_provider()

    def prepare_card(
        self,
        *,
        plugin_id: str,
        envelope: InteractiveCardEnvelope,
        origin: InteractiveCardOrigin,
    ) -> PreparedInteractiveCard:
        if not isinstance(envelope, InteractiveCardEnvelope):
            raise InteractiveCardValidationError("invalid interactive card envelope")
        if not origin.initiator_id.strip() or not origin.message_id.strip():
            raise InteractiveCardValidationError(
                "native interactive cards require a stable initiator and origin message"
            )
        plugin_id = str(plugin_id or "").strip().lower()
        registrations = self._registry()
        action_registrations: dict[str, InteractiveActionRegistration] = {}
        for action in envelope.actions:
            registration = registrations.get(action.action)
            if registration is None or registration.plugin_id != plugin_id:
                raise InteractiveActionRegistrationError(
                    f"interactive action {action.action!r} is not registered by {plugin_id!r}"
                )
            action_registrations[action.action] = registration
        rows: list[
            tuple[
                str,
                InteractiveCardAction,
                InteractiveActionRegistration,
            ]
        ] = []
        seen: set[str] = set()
        for action in envelope.actions:
            instance_id = str(self._id_source() or "").strip()
            if not instance_id or len(instance_id) > 128 or instance_id in seen:
                raise InteractiveCardValidationError(
                    "id source returned an invalid action instance id"
                )
            seen.add(instance_id)
            rows.append((instance_id, action, action_registrations[action.action]))
        now = self._clock()
        self.storage.reserve(
            rows,
            origin=origin,
            expires_at=now + envelope.expires_in_seconds,
            now=now,
        )
        return PreparedInteractiveCard(tuple(row[0] for row in rows))

    def activate_card(self, prepared: PreparedInteractiveCard, *, card_id: str) -> None:
        _bounded_text(card_id, field="card id", limit=512)
        self.storage.activate(
            prepared.action_instance_ids,
            card_id=card_id,
            now=self._clock(),
        )

    def fail_card_delivery(self, prepared: PreparedInteractiveCard) -> None:
        self.storage.fail_delivery(prepared.action_instance_ids, now=self._clock())

    def _record_delivery_failure(self, prepared: PreparedInteractiveCard) -> None:
        """Best-effort terminalization that never masks the delivery error."""

        try:
            self.fail_card_delivery(prepared)
        except Exception:
            logger.warning("Interactive card delivery-failure persistence failed")

    async def deliver_card(
        self,
        *,
        plugin_id: str,
        envelope: InteractiveCardEnvelope,
        origin: InteractiveCardOrigin,
        adapter: Any,
        metadata: Mapping[str, object] | None = None,
    ) -> InteractiveCardDelivery:
        """Deliver a native card or exact fallback through the current adapter."""

        delivery_metadata = dict(metadata) if metadata else None
        if not bool(getattr(adapter, "supports_interactive_cards", False)):
            try:
                result = await adapter.send(
                    chat_id=origin.chat_id,
                    content=envelope.fallback_text,
                    reply_to=origin.message_id or None,
                    metadata=delivery_metadata,
                )
            except Exception as exc:
                raise InteractiveCardDeliveryError(
                    "interactive card fallback could not be delivered"
                ) from exc
            if not getattr(result, "success", False):
                raise InteractiveCardDeliveryError(
                    "interactive card fallback could not be delivered"
                )
            return InteractiveCardDelivery(
                mode="fallback",
                message_id=str(getattr(result, "message_id", None) or ""),
            )

        try:
            prepared = self.prepare_card(
                plugin_id=plugin_id,
                envelope=envelope,
                origin=origin,
            )
        except Exception as exc:
            raise InteractiveCardDeliveryError(
                "interactive card could not be prepared for delivery"
            ) from exc
        try:
            result = await adapter.send_interactive_card(
                chat_id=origin.chat_id,
                envelope=envelope,
                action_instance_ids=prepared.action_instance_ids,
                reply_to=origin.message_id,
                metadata=delivery_metadata,
            )
        except Exception as exc:
            self._record_delivery_failure(prepared)
            raise InteractiveCardDeliveryError(
                "interactive card could not be delivered"
            ) from exc
        if not getattr(result, "success", False) or not getattr(
            result, "message_id", None
        ):
            self._record_delivery_failure(prepared)
            raise InteractiveCardDeliveryError(
                "interactive card could not be delivered"
            )
        card_id = str(result.message_id)
        try:
            self.activate_card(prepared, card_id=card_id)
        except Exception as exc:
            self._record_delivery_failure(prepared)
            try:
                disabled = await adapter.update_interactive_card(
                    chat_id=origin.chat_id,
                    card_id=card_id,
                    result=InteractiveActionResult.retryable_failure(
                        "This confirmation is unavailable. Create a new confirmation."
                    ),
                )
                if not getattr(disabled, "success", False):
                    logger.warning(
                        "Interactive card activation failed and the delivered card disable was rejected"
                    )
            except Exception:
                logger.warning(
                    "Interactive card activation failed and the delivered card could not be disabled"
                )
            raise InteractiveCardDeliveryError(
                "interactive card could not be activated after delivery"
            ) from exc
        return InteractiveCardDelivery(
            mode="native",
            message_id=card_id,
            action_instance_ids=prepared.action_instance_ids,
        )

    async def dispatch(
        self,
        callback: InteractiveActionCallback,
        *,
        gateway_authorize: Callable[[], bool],
    ) -> InteractiveActionResult:
        claim = self.claim_action(
            callback,
            gateway_authorize=gateway_authorize,
        )
        if isinstance(claim, InteractiveActionResult):
            return claim
        return await self.execute_claimed(claim)

    def claim_action(
        self,
        callback: InteractiveActionCallback,
        *,
        gateway_authorize: Callable[[], bool],
    ) -> InteractiveActionResult | ClaimedInteractiveAction:
        """Authorize and atomically claim without running plugin code."""

        if (
            not callback.operator_id.strip()
            or not callback.action_instance_id.strip()
            or len(callback.action_instance_id) > 128
        ):
            return InteractiveActionResult.denied()
        try:
            authorized = gateway_authorize() is True
        except Exception:
            authorized = False
        if not authorized:
            return InteractiveActionResult.denied()

        record = self.storage.get(callback.action_instance_id)
        if record is None:
            return InteractiveActionResult.unknown()
        registration = self._registry().get(record.plugin_action)
        if registration is None:
            return InteractiveActionResult.unknown()
        if (
            record.authorization_policy != "authorized_user"
            or registration.authorization_policy != "authorized_user"
        ) and callback.operator_id != record.initiator_id:
            return InteractiveActionResult.denied()

        claim = self.storage.claim(callback, now=self._clock())
        if claim.status == "processing":
            return InteractiveActionResult.processing()
        if claim.status == "finished" and claim.record is not None:
            outcome = claim.record.outcome
            if outcome in _DEFAULT_RESULT_MESSAGES:
                return InteractiveActionResult(
                    status=cast(InteractiveActionStatus, outcome),
                    user_message=claim.record.user_message,
                )
            return InteractiveActionResult.already_processed()
        if claim.status == "denied":
            return InteractiveActionResult.denied()
        if claim.status == "expired":
            return InteractiveActionResult.expired()
        if claim.status != "claimed" or claim.record is None:
            return InteractiveActionResult.unknown()

        return ClaimedInteractiveAction(
            callback=callback,
            record=claim.record,
            registration=registration,
        )

    async def execute_claimed(
        self,
        claim: ClaimedInteractiveAction,
    ) -> InteractiveActionResult:
        """Run one already-claimed handler and durably record its final state."""

        context = claim.record.handler_context(claim.callback)
        try:
            result = await self._run_handler(claim.registration.handler, context)
        except asyncio.CancelledError:
            try:
                self.mark_unknown_outcome(claim)
            except Exception as exc:
                logger.warning(
                    "Interactive action cancellation persistence failed (%s)",
                    type(exc).__name__,
                )
            raise
        if not self.storage.finish(
            claim.callback.action_instance_id,
            result=result,
            now=self._clock(),
        ):
            raise RuntimeError("interactive action final state was not persisted")
        return result

    def mark_unknown_outcome(self, claim: ClaimedInteractiveAction) -> bool:
        """Terminalize a still-processing claim without overwriting a result."""

        return self.storage.finish(
            claim.callback.action_instance_id,
            result=InteractiveActionResult.unknown_outcome(),
            now=self._clock(),
        )

    async def _run_handler(
        self,
        handler: Callable,
        context: InteractiveActionContext,
    ) -> InteractiveActionResult:
        try:
            if inspect.iscoroutinefunction(handler):
                value = await handler(context)
            else:
                value = await asyncio.to_thread(handler, context)
                if inspect.isawaitable(value):
                    value = await value
            if value is None:
                return InteractiveActionResult.succeeded()
            if isinstance(value, InteractiveActionResult) and value.status in {
                "succeeded",
                "downstream_replay",
                "conflict",
                "retryable_failure",
            }:
                return value
            logger.warning("Interactive action handler returned an invalid result type")
            return InteractiveActionResult.retryable_failure()
        except InteractiveActionConflictError as exc:
            return InteractiveActionResult.conflict(exc.public_message)
        except InteractiveActionRetryableError as exc:
            return InteractiveActionResult.retryable_failure(exc.public_message)
        except Exception as exc:
            logger.warning(
                "Interactive action handler failed (%s)",
                type(exc).__name__,
            )
            return InteractiveActionResult.retryable_failure()


InteractiveCardSender = Callable[..., object]
_CURRENT_CARD_SENDER: ContextVar[InteractiveCardSender | None] = ContextVar(
    "hermes_interactive_card_sender",
    default=None,
)


@contextmanager
def bind_interactive_card_sender(sender: InteractiveCardSender) -> Iterator[None]:
    """Bind the current gateway destination without exposing its identifiers."""

    if not callable(sender):
        raise TypeError("interactive card sender must be callable")
    token = _CURRENT_CARD_SENDER.set(sender)
    try:
        yield
    finally:
        _CURRENT_CARD_SENDER.reset(token)


def send_current_interactive_card(
    *,
    plugin_id: str,
    envelope: InteractiveCardEnvelope,
) -> object:
    """Send through the current gateway turn or fail closed outside one."""

    sender = _CURRENT_CARD_SENDER.get()
    if sender is None:
        raise InteractiveCardUnavailableError(
            "interactive cards are only available during a current gateway turn"
        )
    if not isinstance(envelope, InteractiveCardEnvelope):
        raise InteractiveCardValidationError(
            "send_interactive_card requires an InteractiveCardEnvelope"
        )
    return sender(plugin_id=plugin_id, envelope=envelope)


@dataclass(frozen=True)
class InteractiveActionRegistration:
    """One plugin-owned action handler and its authorization policy."""

    qualified_name: str
    plugin_id: str
    handler: Callable
    authorization_policy: AuthorizationPolicy


def qualify_action_name(plugin_id: str, action_name: str) -> str:
    """Validate and return ``<plugin-id>/<action>``.

    A plugin may pass its local action name or its already-qualified name.  It
    may never register into another plugin's namespace.
    """

    plugin_id = str(plugin_id or "").strip().lower()
    raw = str(action_name or "").strip().lower()
    plugin_parts = plugin_id.split("/")
    if (
        not 1 <= len(plugin_parts) <= 4
        or len(plugin_id) > 255
        or not all(_NAME_RE.fullmatch(part) for part in plugin_parts)
    ):
        raise InteractiveActionRegistrationError("invalid plugin namespace")

    if "/" in raw:
        prefix = f"{plugin_id}/"
        if not raw.startswith(prefix):
            raise InteractiveActionRegistrationError(
                "interactive action namespace must match the registering plugin"
            )
        action = raw[len(prefix) :]
    else:
        action = raw

    if not _NAME_RE.fullmatch(action):
        raise InteractiveActionRegistrationError(
            "interactive action must match [a-z0-9][a-z0-9._-]{0,63}"
        )
    return f"{plugin_id}/{action}"


def build_registration(
    *,
    plugin_id: str,
    action_name: str,
    handler: Callable,
    authorization_policy: str,
) -> InteractiveActionRegistration:
    """Validate one public registration request."""

    if not callable(handler):
        raise InteractiveActionRegistrationError(
            "interactive action handler must be callable"
        )
    if authorization_policy not in {"initiator_only", "authorized_user"}:
        raise InteractiveActionRegistrationError(
            "authorization policy must be initiator_only or authorized_user"
        )
    qualified = qualify_action_name(plugin_id, action_name)
    return InteractiveActionRegistration(
        qualified_name=qualified,
        plugin_id=qualified.rsplit("/", 1)[0],
        handler=handler,
        authorization_policy=authorization_policy,  # type: ignore[arg-type]
    )

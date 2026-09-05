"""AgentRuntime state and usage persistence, mixed into SessionDB."""

import json
import logging
import math
import re
import sqlite3
import time
from typing import Any, List, Mapping, Optional

from agent.runtime_api import RuntimeFailurePhase, RuntimeStateEnvelope, RuntimeUsageReceipt

logger = logging.getLogger(__name__)

# Runtime plugins may persist opaque state, but the host still owns the
# serialization boundary.  These limits keep a malformed or unexpectedly
# large plugin payload from turning one session row into an unbounded blob.
_RUNTIME_STATE_MAX_BYTES = 64 * 1024
_RUNTIME_STATE_MAX_DEPTH = 8
_RUNTIME_STATE_MAX_NODES = 2_048
_RUNTIME_STATE_MAX_KEY_CHARS = 128
_RUNTIME_STATE_MAX_STRING_CHARS = 16 * 1024
_RUNTIME_ID_MAX_CHARS = 128
_RUNTIME_TEXT_MAX_CHARS = 512
_RUNTIME_USAGE_MAX_TOKENS = 10**12
_RUNTIME_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_RUNTIME_CORRELATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_RUNTIME_STATE_AUTH_KEY_RE = re.compile(
    r"(?:^|[_-])(?:auth|access[_-]?token|api[_-]?key|authorization|"
    r"bearer|client[_-]?secret|cookie|credential|id[_-]?token|oauth|"
    r"password|private[_-]?key|refresh[_-]?token|secret|token)(?:$|[_-])",
    re.IGNORECASE,
)

# TRANSITIONAL COMPATIBILITY (remove after all frozen candidate databases have
# crossed the generic-runtime-state migration window): the candidate that
# preceded AgentRuntime v1 stored its Claude SDK resume id directly on the
# ``sessions`` row.  Keep the exact runtime identity and column name in one
# small allowlist.  No other legacy column is eligible for automatic import.
_LEGACY_CLAUDE_RUNTIME_ID = "hermes-claude-agent-sdk"
_LEGACY_CLAUDE_SESSION_COLUMN = "claude_sdk_session_id"
_LEGACY_CLAUDE_STATE_SCHEMA_VERSION = 1
_LEGACY_CLAUDE_STATE_KEY = "external_session_id"


def _validate_runtime_id(runtime_id: Any) -> str:
    """Validate the stable, provider-neutral runtime identity."""
    if not isinstance(runtime_id, str) or not runtime_id:
        raise ValueError("runtime_id must be a non-empty string")
    if (
        len(runtime_id) > _RUNTIME_ID_MAX_CHARS
        or not _RUNTIME_ID_RE.fullmatch(runtime_id)
    ):
        raise ValueError("runtime_id contains unsupported characters")
    return runtime_id


def _validate_runtime_text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    """Validate short receipt metadata without retaining control characters."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{field} must be non-empty")
    if len(value) > _RUNTIME_TEXT_MAX_CHARS:
        raise ValueError(f"{field} is too long")
    if any(ord(char) < 0x20 or 0x7F <= ord(char) < 0xA0 for char in value):
        raise ValueError(f"{field} contains control characters")
    return value


def _runtime_state_value(
    value: Any,
    *,
    path: str = "$",
    depth: int = 0,
    node_count: Optional[list[int]] = None,
) -> Any:
    """Return a plain JSON value after enforcing the state safety contract.

    State is intentionally limited to JSON objects, arrays, strings, finite
    numbers, booleans, and null.  Mapping subclasses and tuples are accepted
    as input because runtime code may expose immutable views, but are copied
    into ordinary JSON-compatible containers before persistence.
    """
    if node_count is None:
        node_count = [0]
    node_count[0] += 1
    if node_count[0] > _RUNTIME_STATE_MAX_NODES:
        raise ValueError("runtime state has too many values")
    if depth > _RUNTIME_STATE_MAX_DEPTH:
        raise ValueError(f"runtime state exceeds maximum depth at {path}")

    if isinstance(value, Mapping):
        if not value:
            return {}
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"runtime state key at {path} must be a string")
            if not key or len(key) > _RUNTIME_STATE_MAX_KEY_CHARS:
                raise ValueError(f"runtime state key at {path} is invalid")
            if _RUNTIME_STATE_AUTH_KEY_RE.search(key):
                raise ValueError(f"runtime state key at {path} is not permitted")
            normalized[key] = _runtime_state_value(
                item,
                path=f"{path}.{key}",
                depth=depth + 1,
                node_count=node_count,
            )
        return normalized

    if isinstance(value, (list, tuple)):
        return [
            _runtime_state_value(
                item,
                path=f"{path}[{index}]",
                depth=depth + 1,
                node_count=node_count,
            )
            for index, item in enumerate(value)
        ]

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"runtime state contains a non-finite number at {path}")
        return value
    if isinstance(value, str):
        if len(value) > _RUNTIME_STATE_MAX_STRING_CHARS:
            raise ValueError(f"runtime state string at {path} is too long")
        return value
    raise ValueError(f"runtime state value at {path} is not JSON-compatible")


def _encode_runtime_state(state: Mapping[str, Any]) -> str:
    """Validate and canonically encode one runtime state object."""
    if not isinstance(state, Mapping):
        raise ValueError("runtime state must be a JSON object")
    normalized = _runtime_state_value(state)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if len(encoded.encode("utf-8")) > _RUNTIME_STATE_MAX_BYTES:
        raise ValueError("runtime state exceeds the maximum encoded size")
    return encoded


def _runtime_state_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys instead of silently choosing one."""
    decoded: dict[str, Any] = {}
    for key, value in pairs:
        if key in decoded:
            raise ValueError("stored runtime state contains duplicate keys")
        decoded[key] = value
    return decoded


def _decode_runtime_state(encoded: str) -> dict[str, Any]:
    """Decode a stored state and reapply the host safety boundary."""
    if not isinstance(encoded, str):
        raise ValueError("stored runtime state is not text")
    if len(encoded.encode("utf-8")) > _RUNTIME_STATE_MAX_BYTES:
        raise ValueError("stored runtime state exceeds the maximum encoded size")
    try:
        decoded = json.loads(encoded, object_pairs_hook=_runtime_state_object_pairs)
    except (TypeError, ValueError) as exc:
        raise ValueError("stored runtime state is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("stored runtime state is not a JSON object")
    # Re-encoding validates depth, forbidden keys, finite numbers, and size.
    _encode_runtime_state(decoded)
    return decoded


def _validate_runtime_usage_receipt(receipt: RuntimeUsageReceipt) -> None:
    """Validate a provider-neutral receipt before it reaches SQLite."""
    if not isinstance(receipt, RuntimeUsageReceipt):
        raise ValueError("receipt must be a RuntimeUsageReceipt")
    _validate_runtime_id(receipt.runtime_id)
    _validate_runtime_text(receipt.provider, "provider")
    _validate_runtime_text(receipt.model, "model")
    for field in ("selected_model", "effective_model", "canonical_model"):
        value = getattr(receipt, field)
        if value is not None:
            _validate_runtime_text(value, field)
    _validate_runtime_text(receipt.model_resolution, "model_resolution")
    _validate_runtime_text(receipt.billing_mode, "billing_mode", allow_empty=True)
    _validate_runtime_text(receipt.cost_status, "cost_status", allow_empty=True)
    for field in (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    ):
        value = getattr(receipt, field)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value > _RUNTIME_USAGE_MAX_TOKENS
        ):
            raise ValueError(f"{field} must be a bounded non-negative integer")
    if not isinstance(receipt.replay_safe, bool):
        raise ValueError("replay_safe must be a boolean")
    if not isinstance(receipt.fallback_used, bool):
        raise ValueError("fallback_used must be a boolean")
    if receipt.failure_phase is not None and not isinstance(
        receipt.failure_phase, RuntimeFailurePhase
    ):
        raise ValueError("failure_phase must be a RuntimeFailurePhase or None")
    if receipt.correlation_id is not None:
        correlation_id = _validate_runtime_text(
            receipt.correlation_id, "correlation_id"
        )
        if not _RUNTIME_CORRELATION_ID_RE.fullmatch(correlation_id):
            raise ValueError("correlation_id contains unsupported characters")


class SessionRuntimeMixin:
    """Provider-neutral runtime state and append-only usage receipts."""

    # ── AgentRuntime v1 state and usage ───────────────────────────────────

    @staticmethod
    def _runtime_state_from_row(row: sqlite3.Row) -> RuntimeStateEnvelope:
        """Validate and decode a runtime state row returned from SQLite."""
        stored_runtime_id = _validate_runtime_id(row["runtime_id"])
        schema_version = row["schema_version"]
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version <= 0
        ):
            raise ValueError("stored runtime state has an invalid schema_version")
        return RuntimeStateEnvelope(
            runtime_id=stored_runtime_id,
            schema_version=schema_version,
            state=_decode_runtime_state(row["state_json"]),
        )

    def _legacy_claude_session_column_exists(self) -> bool:
        """Return whether this database came from the frozen Claude candidate.

        The current host schema deliberately does not declare the candidate's
        provider-specific column.  Inspecting the live table first lets the
        compatibility reader work on both shapes without broadening
        ``SCHEMA_SQL`` or issuing a speculative SELECT against a missing
        column.
        """
        with self._read_ctx() as conn:
            columns = conn.execute('PRAGMA table_info("sessions")').fetchall()
        return any(
            (row["name"] if isinstance(row, sqlite3.Row) else row[1])
            == _LEGACY_CLAUDE_SESSION_COLUMN
            for row in columns
        )

    def _import_legacy_claude_sdk_session_state(
        self, session_id: str
    ) -> Optional[RuntimeStateEnvelope]:
        """Import one frozen-candidate Claude SDK id into generic state.

        This is intentionally a one-way, additive reader: it never updates,
        clears, or removes the legacy column.  The generic row is inserted
        only when the exact Claude runtime key is absent, and the check plus
        insert share one write transaction so concurrent readers cannot
        overwrite a state envelope written by the runtime.

        TRANSITIONAL COMPATIBILITY: remove this method and its constants once
        every supported candidate database has crossed the migration window.
        Until then, plugin removal leaves the imported generic row inert; no
        SDK import or provider policy lives in the host.
        """
        if self.read_only or not self._legacy_claude_session_column_exists():
            return None

        def _do(conn):
            existing = conn.execute(
                """SELECT runtime_id, schema_version, state_json
                     FROM runtime_session_state
                    WHERE session_id = ? AND runtime_id = ?""",
                (session_id, _LEGACY_CLAUDE_RUNTIME_ID),
            ).fetchone()
            if existing is not None:
                return existing

            legacy = conn.execute(
                """SELECT claude_sdk_session_id
                     FROM sessions
                    WHERE id = ?""",
                (session_id,),
            ).fetchone()
            if legacy is None:
                return None
            legacy_session_id = (
                legacy["claude_sdk_session_id"]
                if isinstance(legacy, sqlite3.Row)
                else legacy[0]
            )
            if legacy_session_id is None:
                return None
            try:
                # Treat the old value as an opaque identifier, not as a
                # provider payload.  Printable, bounded text is all the
                # generic state contract needs to preserve safely.
                legacy_session_id = _validate_runtime_text(
                    legacy_session_id,
                    _LEGACY_CLAUDE_SESSION_COLUMN,
                )
                # sqlite3 binds text as UTF-8.  Reject lone surrogates here so
                # malformed legacy bytes cannot escape the compatibility
                # boundary as a driver-level encoding failure.
                legacy_session_id.encode("utf-8")
                state_json = _encode_runtime_state(
                    {_LEGACY_CLAUDE_STATE_KEY: legacy_session_id}
                )
            except (UnicodeEncodeError, ValueError):
                logger.warning(
                    "Skipping transitional Claude SDK state import for "
                    "one session: legacy value is not a bounded opaque id",
                )
                return None

            conn.execute(
                """INSERT INTO runtime_session_state (
                       session_id, runtime_id, schema_version, state_json,
                       updated_at
                   ) VALUES (?, ?, ?, ?, ?)""",
                (
                    session_id,
                    _LEGACY_CLAUDE_RUNTIME_ID,
                    _LEGACY_CLAUDE_STATE_SCHEMA_VERSION,
                    state_json,
                    time.time(),
                ),
            )
            return conn.execute(
                """SELECT runtime_id, schema_version, state_json
                     FROM runtime_session_state
                    WHERE session_id = ? AND runtime_id = ?""",
                (session_id, _LEGACY_CLAUDE_RUNTIME_ID),
            ).fetchone()

        row = self._execute_write(_do)
        return self._runtime_state_from_row(row) if row is not None else None

    def update_runtime_state(
        self, session_id: str, state: RuntimeStateEnvelope
    ) -> None:
        """Upsert one provider-neutral runtime state envelope.

        Runtime state is scoped by the Hermes session and runtime identity,
        not by provider-specific columns on ``sessions``.  The envelope is
        validated and canonically encoded before the transaction starts so
        credentials, non-JSON values, and unbounded payloads cannot enter the
        store.  The session foreign key deliberately requires the host to
        establish the session first; this method never creates or mutates a
        legacy session row as a side effect.
        """
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(state, RuntimeStateEnvelope):
            raise ValueError("state must be a RuntimeStateEnvelope")
        runtime_id = _validate_runtime_id(state.runtime_id)
        if (
            not isinstance(state.schema_version, int)
            or isinstance(state.schema_version, bool)
            or state.schema_version <= 0
        ):
            raise ValueError("state schema_version must be a positive integer")
        state_json = _encode_runtime_state(state.state)

        def _do(conn):
            conn.execute(
                """INSERT INTO runtime_session_state (
                       session_id, runtime_id, schema_version, state_json,
                       updated_at
                   ) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(session_id, runtime_id) DO UPDATE SET
                       schema_version = excluded.schema_version,
                       state_json = excluded.state_json,
                       updated_at = excluded.updated_at""",
                (session_id, runtime_id, state.schema_version, state_json, time.time()),
            )

        self._execute_write(_do)

    def get_runtime_state(
        self, session_id: str, runtime_id: str
    ) -> Optional[RuntimeStateEnvelope]:
        """Read one runtime state envelope, or ``None`` when it is absent.

        The exact transitional Claude runtime identity also performs a
        one-way import from the frozen candidate's optional session column
        when no generic row exists.  Other runtime reads remain pure and the
        current schema never gains that provider-specific column.
        """
        if not isinstance(session_id, str) or not session_id:
            return None
        runtime_id = _validate_runtime_id(runtime_id)
        with self._read_ctx() as conn:
            row = conn.execute(
                """SELECT runtime_id, schema_version, state_json
                     FROM runtime_session_state
                    WHERE session_id = ? AND runtime_id = ?""",
                (session_id, runtime_id),
            ).fetchone()
        if row is None:
            if runtime_id == _LEGACY_CLAUDE_RUNTIME_ID:
                return self._import_legacy_claude_sdk_session_state(session_id)
            return None
        return self._runtime_state_from_row(row)

    def record_runtime_usage_receipt(
        self, session_id: str, receipt: RuntimeUsageReceipt
    ) -> bool:
        """Append one usage receipt and return whether it was newly inserted.

        A non-secret ``correlation_id`` is the sole deduplication key.  A
        repeated correlated receipt is ignored without changing the first
        row; receipts without a correlation id remain independent append-only
        events.  This method intentionally does not update legacy aggregate
        usage columns or ``session_model_usage``.
        """
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        _validate_runtime_usage_receipt(receipt)

        def _do(conn):
            cursor = conn.execute(
                """INSERT OR IGNORE INTO runtime_usage_receipts (
                       session_id, runtime_id, provider, model, selected_model,
                       effective_model, canonical_model, model_resolution,
                       billing_mode, cost_status, input_tokens, output_tokens,
                       cache_read_tokens, cache_write_tokens, reasoning_tokens,
                       replay_safe, correlation_id, fallback_used, failure_phase,
                       recorded_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    receipt.runtime_id,
                    receipt.provider,
                    receipt.model,
                    receipt.selected_model,
                    receipt.effective_model,
                    receipt.canonical_model,
                    receipt.model_resolution,
                    receipt.billing_mode,
                    receipt.cost_status,
                    receipt.input_tokens,
                    receipt.output_tokens,
                    receipt.cache_read_tokens,
                    receipt.cache_write_tokens,
                    receipt.reasoning_tokens,
                    1 if receipt.replay_safe else 0,
                    receipt.correlation_id,
                    1 if receipt.fallback_used else 0,
                    receipt.failure_phase.value
                    if receipt.failure_phase is not None
                    else None,
                    time.time(),
                ),
            )
            return cursor.rowcount == 1

        return bool(self._execute_write(_do))

    def list_runtime_usage_receipts(
        self, session_id: str, runtime_id: Optional[str] = None
    ) -> List[RuntimeUsageReceipt]:
        """Return append-only runtime receipts in insertion order."""
        if not isinstance(session_id, str) or not session_id:
            return []
        params: list[Any] = [session_id]
        runtime_clause = ""
        if runtime_id is not None:
            runtime_id = _validate_runtime_id(runtime_id)
            runtime_clause = " AND runtime_id = ?"
            params.append(runtime_id)
        with self._read_ctx() as conn:
            rows = conn.execute(
                """SELECT runtime_id, provider, model, selected_model,
                              effective_model, canonical_model, model_resolution,
                              billing_mode, cost_status, input_tokens, output_tokens,
                              cache_read_tokens, cache_write_tokens,
                              reasoning_tokens, replay_safe, correlation_id,
                              fallback_used, failure_phase
                         FROM runtime_usage_receipts
                        WHERE session_id = ?"""
                + runtime_clause
                + " ORDER BY id ASC",
                params,
            ).fetchall()
        receipts: List[RuntimeUsageReceipt] = []
        for row in rows:
            receipt = RuntimeUsageReceipt(
                runtime_id=row["runtime_id"],
                provider=row["provider"],
                model=row["model"],
                selected_model=row["selected_model"],
                effective_model=row["effective_model"],
                canonical_model=row["canonical_model"],
                model_resolution=row["model_resolution"],
                billing_mode=row["billing_mode"],
                cost_status=row["cost_status"],
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                cache_read_tokens=row["cache_read_tokens"],
                cache_write_tokens=row["cache_write_tokens"],
                reasoning_tokens=row["reasoning_tokens"],
                replay_safe=bool(row["replay_safe"]),
                correlation_id=row["correlation_id"],
                fallback_used=bool(row["fallback_used"]),
                failure_phase=(
                    RuntimeFailurePhase(row["failure_phase"])
                    if row["failure_phase"] is not None
                    else None
                ),
            )
            _validate_runtime_usage_receipt(receipt)
            receipts.append(receipt)
        return receipts

"""Shared plumbing of the in-memory message reference: transcript lookup and the transcript write guards every writer runs first."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Optional

from persistence import CompressionClosed, CompressionInProgress, ProfileScope, SessionRowMissing, TurnLeaseLost
from persistence.core.identity import JsonValue, MessageId, SessionId
from persistence.conversation.sessions import END_REASON_COMPRESSION

from .._lease_fence import _lease_rows, verify_turn_lease
from ..backend import InMemoryBackend
from .transcript_store import _messages_namespace, _StoredRow, _Transcript

# The scope tables a transcript write changes: the rows and their counters, the session rows (revision, config) and the lease rows its guards reclaim or renew.
_WRITE_TABLES = ("messages", "sessions", "coord_leases", "coord_turn_leases")


class MessageBase:
    """State access and write guards; the message mixins build on it. Each public verb runs under the backend lock, which makes it one atomic step."""

    def __init__(self, backend: InMemoryBackend) -> None:
        self._backend = backend

    def _ns(self, scope: ProfileScope) -> dict[str, Any]:
        return _messages_namespace(self._backend, scope)

    @contextmanager
    def _write_transaction(self, scope: ProfileScope) -> Iterator[None]:
        """One transcript write as the store's write transaction: the block runs on staged copies of the tables a write changes, published as the store's tables when it returns; an exception puts the committed tables back, so a failed write leaves nothing behind."""
        store = self._backend.scope_store(scope)
        committed = {name: store[name] for name in _WRITE_TABLES if name in store}
        store.update((name, copy.deepcopy(table)) for name, table in committed.items())
        try:
            yield
        except BaseException:
            for name in _WRITE_TABLES:
                store.pop(name, None)
            store.update(committed)
            raise

    def _transcript(self, scope: ProfileScope, session_id: SessionId, create: bool = False) -> Optional[_Transcript]:
        transcripts: dict[str, _Transcript] = self._ns(scope)["transcripts"]
        if create:
            return transcripts.setdefault(str(session_id), _Transcript())
        return transcripts.get(str(session_id))

    def _created(self, scope: ProfileScope, session_id: SessionId) -> _Transcript:
        transcript = self._transcript(scope, session_id, create=True)
        assert transcript is not None
        return transcript

    def _sessions(self, scope: ProfileScope) -> dict[str, dict[str, Any]]:
        table: dict[str, dict[str, Any]] = self._backend.scope_store(scope).get("sessions", {})
        return table

    def _row_in_session(self, scope: ProfileScope, session_id: SessionId, message_id: MessageId) -> Optional[_StoredRow]:
        transcript = self._transcript(scope, session_id)
        return None if transcript is None else transcript.find(int(message_id))

    def _refuse_compression_closed(self, scope: ProfileScope, session_id: SessionId) -> None:
        """A session ended by compression takes no new rows; the refusal names the live continuation resolved in the same step."""
        row = self._sessions(scope).get(str(session_id))
        if row is not None and row.get("end_reason") == END_REASON_COMPRESSION:
            from .lineage import compression_tip

            raise CompressionClosed(session_id, SessionId(compression_tip(self._backend, scope, str(session_id))))

    def _require_session(self, scope: ProfileScope, session_id: SessionId) -> None:
        """A row insert needs its session row (the store's foreign key)."""
        if str(session_id) not in self._sessions(scope):
            raise SessionRowMissing(session_id)

    def _refuse_live_leases(self, scope: ProfileScope, session_id: SessionId, kinds: tuple[str, ...]) -> list[tuple[dict[str, Any], str]]:
        """Refuse on a live lease of ``kinds`` held by anyone (compaction first, then turn) and return the stale rows to reclaim once every refusal has passed."""
        stale: list[tuple[dict[str, Any], str]] = []
        for kind, table, key, live in _lease_rows(self._backend, scope, session_id):
            if kind not in kinds:
                continue
            if live and kind == "compaction":
                raise CompressionInProgress(session_id)
            if live:
                raise TurnLeaseLost(session_id, f"Session has an active turn lease; refusing transcript mutation for {str(session_id)!r}")
            stale.append((table, key))
        return stale

    def _write_guards(self, scope: ProfileScope, session_id: SessionId, *, turn_lease_holder: Optional[str] = None, ttl: float = 300.0,
                      reject_live: tuple[str, ...] = (), insert: bool = True) -> None:
        """The store's transcript-write admission (``_check_transcript_write_guards``), refusals before any change: live leases of ``reject_live`` kinds, the caller's own turn lease, a session closed by compression, a missing session row for an insert; then the stale-lease reclaim and the caller's lease renewal."""
        stale = self._refuse_live_leases(scope, session_id, reject_live) if reject_live else []
        verify_turn_lease(self._backend, scope, session_id, turn_lease_holder, ttl, renew=False)
        self._refuse_compression_closed(scope, session_id)
        if insert:
            self._require_session(scope, session_id)
        for table, key in stale:
            del table[key]
        verify_turn_lease(self._backend, scope, session_id, turn_lease_holder, ttl)

    def _bump_transcript_revision(self, scope: ProfileScope, session_id: SessionId) -> None:
        """An in-place rewrite of a row's payload columns moves the owning session's ``transcript_revision``, as the store's update trigger does."""
        row = self._sessions(scope).get(str(session_id))
        if row is not None:
            row["transcript_revision"] = int(row.get("transcript_revision", 0)) + 1

    def _merge_model_config(self, scope: ProfileScope, session_id: SessionId, patch: Mapping[str, JsonValue]) -> None:
        """Merge ``patch`` into the session config (``None`` removes a key); a missing session row is a ``ValueError``, so nothing commits against it."""
        from .sessions import sync_lineage_markers

        row = self._sessions(scope).get(str(session_id))
        if row is None:
            raise ValueError(f"Session not found: {session_id}")
        for key, value in patch.items():
            if value is None:
                row["model_config"].pop(key, None)
            else:
                row["model_config"][key] = value
        sync_lineage_markers(row)

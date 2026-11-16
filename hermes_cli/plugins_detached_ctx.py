"""Capability-gated ``PluginContext`` surface: record durable detached results
(a delegation delivery or a completion note) into a session's transcript.

A detached result is a row written *between* turns: the core records it once
(keyed by ``delegation_id`` or ``admission_id``), it survives compaction
through the compression lineage, and the next turn of the destination session
reloads it as an ordinary ``user`` row. This is the durable half of the
deferred-delivery contract: producers that finish while no turn is live (a
kanban worker, a cron job, a detached delegation) can leave the result where
the session that asked for the work will find it, instead of dropping it in a
side channel the turn loop never reads.

The methods are generic plugin surface: they name no plugin, no team and no
product concept. They need the ``state.detached_result`` capability; without
it they return ``None`` and log once (fail closed, like ``side_question``).
Errors raised by the core are documented, never swallowed: a destination with
a live turn raises ``SessionTurnLeaseLostError`` (retry later — the result is
not recorded), a compression-closed parent raises
``CompressionSessionClosedError`` (the session is gone — do not retry the
same id).

Contract of the keys a caller cannot override: ``metadata`` is the caller's
own namespace, but the keys the core uses for idempotency are *reserved*
(``RESERVED_DELIVERY_KEYS`` / ``RESERVED_NOTE_KEYS``). They are always written
from the dedicated arguments and win over the same key in ``metadata``, so
``metadata`` can never redirect or defeat the "recorded once" guarantee.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

STATE_DETACHED_RESULT_CAPABILITY = "state.detached_result"

# Keys the core reads back to decide "already recorded": the dedicated arguments own them.
RESERVED_DELIVERY_KEYS = frozenset({"delegation_id", "delivery_notice"})
RESERVED_NOTE_KEYS = frozenset({"completion_admission"})


def _require_text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"record detached result requires a non-empty string {what}")
    return value


def _caller_metadata(metadata: Optional[dict[str, Any]], reserved: frozenset) -> dict[str, Any]:
    """The caller's ``metadata`` without the reserved keys (the dedicated arguments win)."""
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError("record detached result requires metadata to be a dict or None")
    dropped = sorted(reserved.intersection(metadata))
    if dropped:
        logger.warning("Ignoring reserved metadata keys %s in a detached result; the dedicated arguments own them",
                       ", ".join(dropped))
    return {key: value for key, value in metadata.items() if key not in reserved}


class PluginDetachedCtxMixin:
    """Mixed into :class:`hermes_cli.plugins.PluginContext` (``manifest``, ``_manager``, ``plugin_id``,
    and ``_granted_in_own_home`` from the capability mixin)."""

    def _detached_result_denied(self) -> None:
        """Fail-closed denial: log once, write nothing."""
        if not getattr(self, "_detached_result_denial_logged", False):
            self._detached_result_denial_logged = True
            logger.info(
                "Plugin %s called a detached-result method without the %s capability; returning None",
                self.plugin_id,
                STATE_DETACHED_RESULT_CAPABILITY,
            )

    def _write_detached(self, session_id: str, write) -> Optional[int]:  # type: ignore[no-untyped-def] -- SessionDB is lazy
        """Resolve *session_id* to the live continuation tip (a parent closed by compression never
        receives rows), then run ``write(db, tip)`` on the shared registry-owned ``SessionDB`` of this
        manager's profile home."""
        from hermes_state_registry import acquire, release

        db = acquire(self._manager.home_path / "state.db")
        try:
            return write(db, db.resolve_resume_session_id(session_id))
        finally:
            release(db)

    def record_delegation_delivery(
        self,
        session_id: str,
        content: str,
        *,
        delegation_id: str,
        delivery_notice: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[int]:
        """Record a detached delegation delivery once, keyed by ``delegation_id`` + ``delivery_notice``.

        Returns the message id of the recorded (or already-recorded) row, or None when the
        ``state.detached_result`` capability is not granted. ``metadata`` is stored with the row; the
        reserved keys ``delegation_id`` and ``delivery_notice`` always come from the arguments. Raises
        ``ValueError`` for an empty ``session_id``, ``content`` or ``delegation_id``; the core's
        ``SessionTurnLeaseLostError`` when the destination session has a live turn (nothing is recorded —
        retry after the turn ends) and ``CompressionSessionClosedError`` when the destination is a closed parent.
        """
        if not self._granted_in_own_home(STATE_DETACHED_RESULT_CAPABILITY):
            return self._detached_result_denied()
        _require_text(session_id, "session_id")
        _require_text(content, "content")
        _require_text(delegation_id, "delegation_id")
        if not isinstance(delivery_notice, str):
            raise ValueError("record detached result requires delivery_notice to be a string")
        meta = {**_caller_metadata(metadata, RESERVED_DELIVERY_KEYS),
                "delegation_id": delegation_id, "delivery_notice": delivery_notice}
        return self._write_detached(session_id, lambda db, tip: db.append_delegation_delivery(tip, content, meta))

    def record_completion_note(
        self,
        session_id: str,
        content: str,
        *,
        admission_id: Optional[str] = None,
        display_kind: str = "internal_notification",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[int]:
        """Record an unviewed completion's note, once per ``admission_id``.

        Returns the message id of the recorded (or already-recorded) row, or None when the
        ``state.detached_result`` capability is not granted. ``admission_id`` is the idempotency key:
        the same string returns the existing row. ``None`` is kept as a real null (stored as JSON
        ``null``, never the string ``"None"``) and has no key to match, so each call records its own
        row; pass a stable id whenever a retry is possible. Any other non-string, or an empty string,
        raises ``ValueError``. The reserved key ``completion_admission`` always comes from
        ``admission_id``. Also raises ``ValueError`` for an empty ``session_id``, ``content`` or
        ``display_kind``; the core's ``SessionTurnLeaseLostError`` when the destination session has a
        live turn (nothing is recorded — retry after the turn ends) and ``CompressionSessionClosedError``
        when the destination is a closed parent. ``display_kind`` selects how the row presents; prefer
        the default unless the core documents a kind for your producer.
        """
        if not self._granted_in_own_home(STATE_DETACHED_RESULT_CAPABILITY):
            return self._detached_result_denied()
        _require_text(session_id, "session_id")
        _require_text(content, "content")
        _require_text(display_kind, "display_kind")
        if admission_id is not None:
            _require_text(admission_id, "admission_id")
        meta = _caller_metadata(metadata, RESERVED_NOTE_KEYS)
        return self._write_detached(
            session_id, lambda db, tip: db.append_completion_note(tip, admission_id, content, display_kind, meta))

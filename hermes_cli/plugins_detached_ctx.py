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
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from hermes_cli.plugin_capabilities import plugin_capability_granted
from hermes_cli.plugins_loader import _plugin_home_scope

logger = logging.getLogger(__name__)

STATE_DETACHED_RESULT_CAPABILITY = "state.detached_result"


class PluginDetachedCtxMixin:
    """Mixed into :class:`hermes_cli.plugins.PluginContext` (``manifest``, ``_manager``, ``plugin_id``)."""

    def _detached_result_granted(self, capability: str) -> bool:
        """Consent read from THIS manager's profile home: plugins call from their own threads,
        where the active profile may be the launch profile's."""
        try:
            from hermes_cli.config import load_config

            with _plugin_home_scope(self._manager.home_path):
                cfg = load_config() or {}
        except Exception:  # health: allow BLE001 -- fail closed: unreadable consent state never grants
            return False
        return plugin_capability_granted(self.plugin_id, capability, config=cfg)

    def _detached_result_denied(self) -> Optional[int]:
        """Fail-closed denial: log once, return None, write nothing."""
        if not getattr(self, "_detached_result_denial_logged", False):
            self._detached_result_denial_logged = True
            logger.info(
                "Plugin %s called a detached-result method without the %s capability; returning None",
                self.plugin_id,
                STATE_DETACHED_RESULT_CAPABILITY,
            )
        return None

    def _record_detached(
        self,
        session_id: str,
        content: str,
        *,
        kind: str,
        delegation_id: Optional[str] = None,
        admission_id: Optional[str] = None,
        delivery_notice: str = "",
        display_kind: str = "internal_notification",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[int]:
        """Shared write path for both detached-result kinds.

        Resolves *session_id* to the live continuation tip first (a parent closed by
        compression never receives rows), then writes exactly once through the shared
        registry-owned ``SessionDB`` of this manager's profile home.
        """
        if not self._detached_result_granted(STATE_DETACHED_RESULT_CAPABILITY):
            return self._detached_result_denied()
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("record detached result requires a non-empty session_id")
        if not isinstance(content, str) or not content:
            raise ValueError("record detached result requires non-empty string content")
        from hermes_state_registry import acquire, release

        db = acquire(self._manager.home_path / "state.db")
        try:
            target = db.resolve_resume_session_id(session_id)
            if kind == "delegation_delivery":
                if not delegation_id:
                    raise ValueError(
                        "delegation delivery requires a stable delegation_id"
                    )
                meta = {
                    "delegation_id": delegation_id,
                    "delivery_notice": delivery_notice,
                }
                meta.update(metadata or {})
                return db.append_delegation_delivery(target, content, meta)
            meta = dict(metadata or {})
            return db.append_completion_note(
                target, str(admission_id), content, display_kind, meta
            )
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
        """Record a detached delegation delivery once, keyed by ``delegation_id``.

        Returns the message id of the recorded (or already-recorded) row, or None
        when the ``state.detached_result`` capability is not granted. Raises
        ``ValueError`` for a missing ``delegation_id``; the core's
        ``SessionTurnLeaseLostError`` when the destination session has a live turn
        (nothing is recorded — retry after the turn ends) and
        ``CompressionSessionClosedError`` when the destination is a closed parent.
        """
        return self._record_detached(
            session_id,
            content,
            kind="delegation_delivery",
            delegation_id=delegation_id,
            delivery_notice=delivery_notice,
            metadata=metadata,
        )

    def record_completion_note(
        self,
        session_id: str,
        content: str,
        *,
        admission_id: str,
        display_kind: str = "internal_notification",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[int]:
        """Record an unviewed completion's note once, keyed by ``admission_id``.

        Returns the message id of the recorded (or already-recorded) row, or None
        when the ``state.detached_result`` capability is not granted. Raises
        ``ValueError`` for a missing ``admission_id``; the core's
        ``SessionTurnLeaseLostError`` when the destination session has a live turn
        (nothing is recorded — retry after the turn ends) and
        ``CompressionSessionClosedError`` when the destination is a closed parent.
        ``display_kind`` selects how the row presents; callers should prefer the
        default unless the core documents a kind for their producer.
        """
        return self._record_detached(
            session_id,
            content,
            kind="completion_note",
            admission_id=admission_id,
            display_kind=display_kind,
            metadata=metadata,
        )

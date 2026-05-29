"""AgentOS session lifecycle for Hermes conversations."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .agentos_bridge import KynverAgentOSClient, KynverAgentOSError

logger = logging.getLogger(__name__)


class KynverSessionManager:
    """Open, log, and close AgentOS sessions with provenance."""

    def __init__(self, client: KynverAgentOSClient):
        self._client = client
        self.agentos_session_id: str = ""
        self._channel: str = ""
        self._model: str = ""

    def open_session(
        self,
        *,
        channel: str,
        model: str = "",
        hermes_session_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        body: Dict[str, Any] = {"channel": channel or "hermes"}
        if model:
            body["model"] = model
        meta = dict(metadata or {})
        if hermes_session_id:
            meta.setdefault("hermesSessionId", hermes_session_id)
        if meta:
            body["metadata"] = meta
        try:
            result = self._client.post("/sessions", body)
            if isinstance(result, dict):
                self.agentos_session_id = str(
                    result.get("id") or result.get("sessionId") or ""
                )
            self._channel = channel
            self._model = model
            return self.agentos_session_id
        except KynverAgentOSError as exc:
            logger.debug("Kynver session open failed: %s", exc)
            return ""

    def log_event(
        self,
        summary: str,
        *,
        event_type: str = "note",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.agentos_session_id or not (summary or "").strip():
            return
        event: Dict[str, Any] = {
            "type": event_type,
            "summary": summary.strip()[:2000],
        }
        if details:
            event["details"] = details
        try:
            self._client.post(
                f"/sessions/{self.agentos_session_id}/events",
                {"event": event},
            )
        except KynverAgentOSError:
            logger.debug("Kynver session event log failed", exc_info=True)

    def close_session(
        self,
        summary: str = "",
        *,
        topics: Optional[List[str]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not self.agentos_session_id:
            return
        body: Dict[str, Any] = {}
        if summary:
            body["summary"] = summary.strip()[:4000]
        if topics:
            body["topicsWorked"] = topics
        if events:
            body["events"] = events
        try:
            self._client.patch(f"/sessions/{self.agentos_session_id}", body)
        except KynverAgentOSError:
            logger.debug("Kynver session close failed", exc_info=True)
        finally:
            self.agentos_session_id = ""

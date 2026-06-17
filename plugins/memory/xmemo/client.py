"""Synchronous REST client for XMemo.

Deliberately lightweight: uses ``httpx.Client`` directly instead of the async
``memory_manager.client.RemoteMemoryManager`` so Hermes does not inherit the
full Memory OS server dependency tree.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class XMemoClientError(Exception):
    """Raised when an XMemo API call fails."""

    def __init__(self, message: str, status_code: int = 0, response_body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class XMemoClient:
    """Synchronous XMemo REST client."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        agent_id: str = "hermes",
        agent_instance_id: str = "",
        timeout: float = 5.0,
        transport: Optional[Any] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.agent_id = agent_id
        self.agent_instance_id = agent_instance_id
        self.timeout = timeout

        self.headers: Dict[str, str] = {"X-API-Key": api_key}
        if agent_id:
            self.headers["X-Memory-OS-Agent-ID"] = agent_id
        if agent_instance_id:
            self.headers["X-Memory-OS-Agent-Instance-ID"] = agent_instance_id

        client_kwargs: Dict[str, Any] = {
            "base_url": self.base_url,
            "headers": self.headers,
            "timeout": httpx.Timeout(timeout),
        }
        if transport is not None:
            client_kwargs["transport"] = transport
        self._client = httpx.Client(**client_kwargs)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make a synchronous request and return parsed JSON."""
        url = f"{self.base_url}{path}"
        try:
            response = self._client.request(
                method=method,
                url=url,
                params=params,
                json=json_body,
            )
            response.raise_for_status()
            if response.status_code == 204 or not response.content:
                return {}
            return response.json()
        except httpx.HTTPStatusError as exc:
            body = None
            try:
                body = exc.response.json()
            except Exception:
                body = exc.response.text
            logger.debug("XMemo API error: %s %s -> %s: %s", method, path, exc.response.status_code, body)
            raise XMemoClientError(
                f"XMemo API error {exc.response.status_code}: {body}",
                status_code=exc.response.status_code,
                response_body=body,
            ) from exc
        except Exception as exc:
            logger.debug("XMemo request failed: %s %s -> %s", method, path, exc)
            raise XMemoClientError(f"XMemo request failed: {exc}") from exc

    def health(self) -> Dict[str, Any]:
        """Check service health."""
        return self._request("GET", "/health")

    def recall_context(
        self,
        query: str,
        *,
        bucket: str = "work",
        scope: str = "hermes/default",
        max_items: int = 5,
        max_tokens: int = 900,
        memory_type: str = "auto",
        prefer_working: bool = True,
    ) -> Dict[str, Any]:
        """Build a bounded context pack from XMemo memories."""
        return self._request(
            "POST",
            "/v1/recall/context",
            json_body={
                "query": query,
                "bucket": bucket,
                "scope": scope,
                "max_items": max_items,
                "max_tokens": max_tokens,
                "memory_type": memory_type,
                "prefer_working": prefer_working,
            },
        )

    def search(
        self,
        query: str,
        *,
        bucket: str = "work",
        scope: str = "hermes/default",
        memory_type: str = "%",
        limit: int = 5,
        explain: bool = False,
        prefer_working: bool = False,
    ) -> List[Dict[str, Any]]:
        """Semantic search over XMemo memories."""
        result = self._request(
            "GET",
            "/v1/memories/search",
            params={
                "query": query,
                "bucket": bucket,
                "scope": scope,
                "memory_type": memory_type,
                "limit": limit,
                "explain": explain,
                "prefer_working": prefer_working,
            },
        )
        if isinstance(result, dict):
            return result.get("results", []) or []
        if isinstance(result, list):
            return result
        return []

    def remember(
        self,
        content: str,
        path: str,
        *,
        bucket: str = "work",
        scope: str = "hermes/default",
        memory_type: str = "semantic",
        importance: Optional[float] = None,
        confidence: Optional[float] = None,
        dedupe: bool = True,
        semantic_key: str = "",
    ) -> Dict[str, Any]:
        """Save a durable fact to XMemo."""
        payload: Dict[str, Any] = {
            "content": content,
            "path": path,
            "bucket": bucket,
            "scope": scope,
            "memory_type": memory_type,
            "dedupe": dedupe,
        }
        if importance is not None:
            payload["importance"] = importance
        if confidence is not None:
            payload["confidence"] = confidence
        if semantic_key:
            payload["semantic_key"] = semantic_key
        return self._request("POST", "/v1/remember", json_body=payload)

    def update_state(
        self,
        *,
        state_key: str = "active_task",
        content: str = "",
        current_task: str = "",
        next_action: str = "",
        blocked_reason: str = "",
        bucket: str = "work",
        scope: str = "hermes/default",
        ttl_seconds: int = 86400,
    ) -> Dict[str, Any]:
        """Persist active working state with TTL."""
        payload: Dict[str, Any] = {
            "state_key": state_key,
            "bucket": bucket,
            "scope": scope,
            "ttl_seconds": ttl_seconds,
        }
        if content:
            payload["content"] = content
        if current_task:
            payload["current_task"] = current_task
        if next_action:
            payload["next_action"] = next_action
        if blocked_reason:
            payload["blocked_reason"] = blocked_reason
        return self._request("POST", "/v1/update_state", json_body=payload)

    def record_event(
        self,
        content: str,
        *,
        event_type: str = "event",
        bucket: str = "work",
        scope: str = "hermes/default",
        session_id: str = "",
    ) -> Dict[str, Any]:
        """Append a timeline event."""
        payload: Dict[str, Any] = {
            "content": content,
            "event_type": event_type,
            "bucket": bucket,
            "scope": scope,
        }
        if session_id:
            payload["session_id"] = session_id
        return self._request("POST", "/v1/timeline/events", json_body=payload)

    def create_restart_snapshot(
        self,
        *,
        session_id: str = "",
        bucket: str = "work",
        scope: str = "hermes/default",
        state_key: str = "active_task",
    ) -> Dict[str, Any]:
        """Capture a restart snapshot before handoff or shutdown."""
        payload: Dict[str, Any] = {
            "bucket": bucket,
            "scope": scope,
            "state_key": state_key,
        }
        if session_id:
            payload["session_id"] = session_id
        return self._request("POST", "/v1/restart/snapshot", json_body=payload)

    def create_reminder(
        self,
        content: str,
        *,
        bucket: str = "work",
        scope: str = "hermes/default",
        due_at: str = "",
        session_id: str = "",
    ) -> Dict[str, Any]:
        """Create a TODO/action item to revisit later."""
        payload: Dict[str, Any] = {
            "content": content,
            "bucket": bucket,
            "scope": scope,
        }
        if due_at:
            payload["due_at"] = due_at
        if session_id:
            payload["session_id"] = session_id
        return self._request("POST", "/v1/reminders", json_body=payload)

    def list_reminders(
        self,
        *,
        bucket: str = "work",
        scope: str = "hermes/default",
        item_status: str = "open",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List open or completed TODO items."""
        result = self._request(
            "GET",
            "/v1/reminders",
            params={
                "bucket": bucket,
                "scope": scope,
                "item_status": item_status,
                "limit": limit,
            },
        )
        if isinstance(result, dict):
            return result.get("items", []) or result.get("reminders", []) or []
        if isinstance(result, list):
            return result
        return []

    def complete_reminder(
        self,
        todo_id: str,
        *,
        bucket: str = "work",
        scope: str = "hermes/default",
        note: str = "",
    ) -> Dict[str, Any]:
        """Mark a TODO item as completed."""
        payload: Dict[str, Any] = {"bucket": bucket, "scope": scope}
        if note:
            payload["note"] = note
        return self._request(
            "POST", f"/v1/reminders/{todo_id}/complete", json_body=payload
        )

    def mark_used(
        self,
        memory_id: str,
        *,
        context: str = "",
        action: str = "used",
        usage_tracking_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record that a recalled memory was used in the answer.

        Payload matches Memory OS MemoryUsageRequest: only usage_tracking_id,
        action, context, and metadata are accepted (extra="forbid").
        """
        payload: Dict[str, Any] = {"action": action}
        if context:
            payload["context"] = context
        if usage_tracking_id:
            payload["usage_tracking_id"] = usage_tracking_id
        if metadata:
            payload["metadata"] = metadata
        return self._request(
            "POST", f"/v1/memories/{memory_id}/usage", json_body=payload
        )

    def forget(
        self,
        memory_id: str,
        *,
        bucket: str = "work",
        scope: str = "hermes/default",
        reason: str = "",
    ) -> Dict[str, Any]:
        """Delete a memory by exact id."""
        payload: Dict[str, Any] = {"bucket": bucket, "scope": scope}
        if reason:
            payload["reason"] = reason
        return self._request(
            "POST", f"/v1/memories/{memory_id}/forget", json_body=payload
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        try:
            self._client.close()
        except Exception as exc:
            logger.debug("XMemo client close failed: %s", exc)

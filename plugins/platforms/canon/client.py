"""Canon REST and SSE transport for the Canon platform plugin."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from gateway.platforms.base import safe_url_for_log, _ssrf_redirect_guard

from plugins.platforms.canon.constants import (
    DEFAULT_BASE_URL,
    DEFAULT_HISTORY_LIMIT,
    DEFAULT_STREAM_URL,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_MEDIA_BYTES,
)
from plugins.platforms.canon.models import CanonApiError, CanonStreamFrame

class CanonHttpClient:
    """Small async client for the Canon agent REST and SSE APIs."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        stream_url: str = DEFAULT_STREAM_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.stream_url = stream_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            event_hooks={"response": [_ssrf_redirect_guard]},
        )

    def _headers(self, *, accept: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if accept:
            headers["Accept"] = accept
        return headers

    async def close(self) -> None:
        await self._client.aclose()

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
    ) -> Any:
        res = await self._client.request(
            method,
            f"{self.base_url}{path}",
            headers=self._headers(),
            params=params,
            json=json_body,
        )
        if res.status_code >= 400:
            raise CanonApiError(res.status_code, res.text)
        if not res.content:
            return {}
        return res.json()

    async def get_me(self) -> dict[str, Any]:
        data = await self._request_json("GET", "/agents/me")
        return data if isinstance(data, dict) else {}

    async def get_conversations(self) -> list[dict[str, Any]]:
        data = await self._request_json("GET", "/conversations")
        conversations = data.get("conversations") if isinstance(data, dict) else data
        return conversations if isinstance(conversations, list) else []

    async def get_messages(
        self, conversation_id: str, *, limit: int = DEFAULT_HISTORY_LIMIT
    ) -> list[dict[str, Any]]:
        data = await self._request_json(
            "GET",
            f"/conversations/{conversation_id}/messages",
            params={"limit": str(limit)},
        )
        messages = data.get("messages") if isinstance(data, dict) else data
        return messages if isinstance(messages, list) else []

    async def send_message(
        self,
        conversation_id: str,
        text: str,
        *,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "conversationId": conversation_id,
            "text": text,
        }
        if options:
            body.update(options)
        if reply_to:
            body["replyTo"] = reply_to
        if metadata:
            body["metadata"] = metadata

        data = await self._request_json("POST", "/messages/send", json_body=body)
        return data if isinstance(data, dict) else {}

    async def upload_media(
        self,
        conversation_id: str,
        data: str,
        mime_type: str,
        *,
        file_name: Optional[str] = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "conversationId": conversation_id,
            "data": data,
            "mimeType": mime_type,
        }
        if file_name:
            body["fileName"] = file_name
        result = await self._request_json("POST", "/media/upload", json_body=body)
        return result if isinstance(result, dict) else {}

    async def download_media(self, url: str) -> tuple[bytes, Optional[str]]:
        from tools.url_safety import is_safe_url

        if not is_safe_url(url):
            raise ValueError(
                f"Blocked unsafe URL (SSRF protection): {safe_url_for_log(url)}"
            )

        res = await self._client.get(
            url,
            headers={"User-Agent": "HermesAgent/CanonPlatform"},
            follow_redirects=True,
        )
        if res.status_code >= 400:
            raise CanonApiError(res.status_code, res.text)
        if len(res.content) > MAX_MEDIA_BYTES:
            raise ValueError("Canon media attachment exceeds 10MB")
        return res.content, res.headers.get("content-type")

    async def set_typing(
        self,
        conversation_id: str,
        typing: bool,
        status: Optional[str] = None,
    ) -> None:
        body: dict[str, Any] = {
            "conversationId": conversation_id,
            "typing": typing,
        }
        if status in {"typing", "thinking"}:
            body["status"] = status
        await self._request_json("POST", "/typing", json_body=body)

    async def update_runtime_status(
        self,
        *,
        runtime: str = "hermes",
        host_mode: bool = False,
        runtime_descriptor: Optional[dict[str, Any]] = None,
    ) -> None:
        body: dict[str, Any] = {
            "runtime": runtime,
            "hostMode": host_mode,
        }
        if runtime_descriptor:
            body["runtimeDescriptor"] = runtime_descriptor
        await self._request_json("POST", "/runtime/status", json_body=body)

    async def update_runtime_turn(
        self,
        conversation_id: str,
        *,
        state: str,
        turn_id: Optional[str] = None,
        queue_depth: int = 0,
        active_message_ids: Optional[list[str]] = None,
        capabilities: Optional[dict[str, Any]] = None,
        opened_at: Optional[int] = None,
        turn_updated_at: Optional[int] = None,
    ) -> None:
        body: dict[str, Any] = {
            "conversationId": conversation_id,
            "state": state,
            "queueDepth": max(0, int(queue_depth)),
        }
        if turn_id is not None:
            body["turnId"] = turn_id
        if active_message_ids:
            body["activeMessageIds"] = active_message_ids
        if capabilities:
            body["capabilities"] = capabilities
        if opened_at is not None:
            body["openedAt"] = opened_at
        if turn_updated_at is not None:
            body["turnUpdatedAt"] = turn_updated_at
        await self._request_json("POST", "/runtime/turn", json_body=body)

    async def set_streaming(
        self,
        conversation_id: str,
        *,
        text: str,
        status: str = "streaming",
        turn_id: Optional[str] = None,
    ) -> None:
        body: dict[str, Any] = {
            "conversationId": conversation_id,
            "text": text,
            "status": status,
        }
        if turn_id:
            body["messageId"] = turn_id
            body["turnId"] = turn_id
        await self._request_json("POST", "/streaming", json_body=body)

    async def clear_streaming(self, conversation_id: str) -> None:
        await self._request_json(
            "POST",
            "/streaming",
            json_body={"conversationId": conversation_id, "streaming": False},
        )

    async def update_message_disposition(
        self,
        conversation_id: str,
        message_id: str,
        inbound_disposition: str,
    ) -> None:
        await self._request_json(
            "PATCH",
            f"/conversations/{conversation_id}/messages/{message_id}/disposition",
            json_body={"inboundDisposition": inbound_disposition},
        )

    async def mark_as_read(self, conversation_id: str) -> None:
        await self._request_json("POST", f"/conversations/{conversation_id}/read")

    async def create_runtime_input_request(
        self,
        conversation_id: str,
        *,
        input_id: str,
        kind: str,
        expires_at: int,
        title: Optional[str] = None,
        prompt: Optional[str] = None,
        choices: Optional[list[dict[str, Any]]] = None,
        questions: Optional[list[dict[str, Any]]] = None,
        response_user_id: Optional[str] = None,
        native: Optional[dict[str, Any]] = None,
        sensitive: Optional[bool] = None,
        turn_id: Optional[str] = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "conversationId": conversation_id,
            "inputId": input_id,
            "kind": kind,
            "expiresAt": expires_at,
        }
        if title:
            body["title"] = title
        if prompt:
            body["prompt"] = prompt
        if choices:
            body["choices"] = choices
        if questions:
            body["questions"] = questions
        if response_user_id:
            body["responseUserId"] = response_user_id
        if native:
            body["native"] = native
        if sensitive is not None:
            body["sensitive"] = bool(sensitive)
        if turn_id:
            body["turnId"] = turn_id
        data = await self._request_json("POST", "/runtime-input/request", json_body=body)
        return data if isinstance(data, dict) else {}

    async def consume_runtime_input_response(
        self,
        conversation_id: str,
        input_id: str,
        *,
        cancel: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "conversationId": conversation_id,
            "inputId": input_id,
        }
        if cancel:
            body["cancel"] = True
        data = await self._request_json("POST", "/runtime-input/consume", json_body=body)
        return data if isinstance(data, dict) else {}

    async def create_runtime_card_request(
        self,
        conversation_id: str,
        *,
        card: dict[str, Any],
        card_id: Optional[str] = None,
        expires_at: Optional[int] = None,
        response_user_id: Optional[str] = None,
        runtime_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        native: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "conversationId": conversation_id,
            "card": card,
        }
        if card_id:
            body["cardId"] = card_id
        if expires_at is not None:
            body["expiresAt"] = expires_at
        if response_user_id:
            body["responseUserId"] = response_user_id
        if runtime_id:
            body["runtimeId"] = runtime_id
        if turn_id:
            body["turnId"] = turn_id
        if native:
            body["native"] = native
        data = await self._request_json("POST", "/runtime-card/request", json_body=body)
        return data if isinstance(data, dict) else {}

    async def consume_runtime_card_response(
        self,
        conversation_id: str,
        card_id: str,
        *,
        cancel: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "conversationId": conversation_id,
            "cardId": card_id,
        }
        if cancel:
            body["cancel"] = True
        data = await self._request_json("POST", "/runtime-card/consume", json_body=body)
        return data if isinstance(data, dict) else {}

    async def create_runtime_approval_request(
        self,
        conversation_id: str,
        *,
        approval_id: str,
        tool_name: str,
        tool_summary: str,
        expires_at: int,
        details: Optional[list[dict[str, Any]]] = None,
        native: Optional[dict[str, Any]] = None,
        allow_session_rule: bool = True,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "conversationId": conversation_id,
            "approvalId": approval_id,
            "toolName": tool_name,
            "toolSummary": tool_summary,
            "expiresAt": expires_at,
            "category": "command",
            "risk": "high",
            "riskLevel": "destructive",
            "allowSessionRule": allow_session_rule,
        }
        if details:
            body["details"] = details
        if native:
            body["native"] = native
        data = await self._request_json("POST", "/runtime-approval/request", json_body=body)
        return data if isinstance(data, dict) else {}

    async def consume_runtime_approval_response(
        self,
        conversation_id: str,
        approval_id: str,
        *,
        cancel: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "conversationId": conversation_id,
            "approvalId": approval_id,
        }
        if cancel:
            body["cancel"] = True
        data = await self._request_json("POST", "/runtime-approval/consume", json_body=body)
        return data if isinstance(data, dict) else {}

    async def consume_runtime_signal(self, conversation_id: str) -> dict[str, Any]:
        data = await self._request_json(
            "POST",
            "/runtime/signal/consume",
            json_body={"conversationId": conversation_id},
        )
        return data if isinstance(data, dict) else {}

    async def stream_events(
        self,
        *,
        last_event_id: Optional[str] = None,
    ) -> AsyncIterator[CanonStreamFrame]:
        url = f"{self.stream_url}/agents/stream"
        headers = self._headers(accept="text/event-stream")
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id

        async with self._client.stream(
            "GET",
            url,
            headers=headers,
            params={"events": "messages"},
            timeout=None,
        ) as res:
            if res.status_code >= 400:
                body = await res.aread()
                raise CanonApiError(
                    res.status_code, body.decode("utf-8", errors="replace")
                )

            buffer = ""
            async for chunk in res.aiter_text():
                if not chunk:
                    continue
                buffer += chunk.replace("\r\n", "\n").replace("\r", "\n")
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    parsed = _parse_sse_frame(frame)
                    if parsed is not None:
                        yield parsed




def _parse_sse_frame(frame: str) -> Optional[CanonStreamFrame]:
    event: Optional[str] = None
    event_id: Optional[str] = None
    data_lines: list[str] = []

    for line in frame.split("\n"):
        if not line or line.startswith(":"):
            continue
        if line.startswith("id:"):
            event_id = line[3:].strip()
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())

    if not event or not data_lines:
        return None

    raw_data = "\n".join(data_lines)
    try:
        data: Any = json.loads(raw_data)
    except json.JSONDecodeError:
        data = raw_data
    return CanonStreamFrame(event=event, data=data, event_id=event_id)




def _safe_error(exc: Exception) -> str:
    if isinstance(exc, CanonApiError):
        return str(exc)
    return str(exc) or exc.__class__.__name__


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, CanonApiError):
        return exc.retryable
    return isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ),
    )



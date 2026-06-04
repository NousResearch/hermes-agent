"""
Canon platform adapter for Hermes Agent.

This bundled plugin connects Hermes to Canon agent conversations using the
same public surface as Canon's TypeScript SDK:

* REST endpoints for identity, conversation discovery, sends, and typing.
* Server-sent events from /agents/stream?events=messages for inbound messages.

The adapter is intentionally dependency-light: Hermes already depends on
httpx, so we do not need a Node sidecar or a runtime dependency on
@canonmsg/* packages.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Deque, Dict, Optional

import httpx

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_bytes,
    cache_video_from_bytes,
    safe_url_for_log,
    _ssrf_redirect_guard,
)
from gateway.session import build_session_key

logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = "https://api-6m6mlelskq-uc.a.run.app"
DEFAULT_STREAM_URL = "https://canon-agent-stream-195218560334.us-central1.run.app"
DEFAULT_HISTORY_LIMIT = 50
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_SEEN_MESSAGE_IDS = 1024
MAX_MEDIA_BYTES = 10 * 1024 * 1024
RUNTIME_STATUS_INTERVAL_SECONDS = 30.0
RUNTIME_SIGNAL_POLL_SECONDS = 1.0
RUNTIME_INPUT_POLL_SECONDS = 1.0
RUNTIME_APPROVAL_POLL_SECONDS = 1.0
RUNTIME_HITL_MAX_TIMEOUT_SECONDS = 30 * 60
FINAL_MESSAGE_HANDOFF_SECONDS = 0.75
REGISTRATION_POLL_INTERVAL_SECONDS = 3.0
REGISTRATION_TIMEOUT_SECONDS = 5 * 60.0

AUDIO_EXTS = {".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm", ".flac"}
IMAGE_EXTS = {".gif", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_EXTS = {".avi", ".mkv", ".mov", ".mp4", ".webm", ".3gp"}

TURN_COMPLETE_METADATA = {
    "turnSemantics": "turn_complete",
    "turnComplete": True,
}
CANON_AGENTS_JSON_BOOTSTRAP_ENV = "CANON_AGENTS_JSON_BOOTSTRAP"

CONTROL_METADATA_TYPES = {
    "approval_reply",
    "approval_outcome",
    "runtime_input_reply",
    "runtime_input_outcome",
}

CANON_HERMES_RUNTIME_DESCRIPTOR: dict[str, Any] = {
    "coreControls": [],
    "runtimeControls": [],
    "commands": [],
    "actions": [
        {
            "id": "stop",
            "label": "Stop",
            "description": "Interrupt the current Hermes turn.",
            "aliases": ["stop"],
            "category": "turn",
            "placements": ["composer_slash", "command_palette"],
            "availability": ["busy"],
            "dispatch": {"kind": "signal", "signal": "interrupt"},
        },
        {
            "id": "stop-and-clear-queue",
            "label": "Stop & clear queue",
            "description": "Interrupt the current Hermes turn and drop queued Canon prompts.",
            "aliases": ["stop-clear", "clear-queue"],
            "category": "turn",
            "placements": ["composer_slash", "command_palette", "session_strip"],
            "availability": ["busy_with_queue"],
            "dispatch": {"kind": "signal", "signal": "stop_and_drop"},
        },
        {
            "id": "new-session",
            "label": "New session",
            "description": "Start a fresh Hermes session inside this Canon chat.",
            "aliases": ["new"],
            "category": "session",
            "placements": ["composer_slash", "command_palette", "session_strip"],
            "availability": ["always"],
            "dispatch": {"kind": "signal", "signal": "new_session"},
        },
    ],
    "supportsInterrupt": True,
    "supportsInputInterrupt": True,
    "streamingTextMode": "delta",
}


class CanonApiError(Exception):
    """HTTP error from Canon's agent API."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body.strip()
        super().__init__(f"Canon API returned {status_code}: {self.body[:500]}")

    @property
    def retryable(self) -> bool:
        return self.status_code in {408, 425, 429} or self.status_code >= 500


@dataclass
class CanonStreamFrame:
    event: str
    data: Any
    event_id: Optional[str] = None


@dataclass
class CanonResolvedAgent:
    api_key: str = ""
    profile: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    base_url: str = ""
    stream_url: str = ""


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


class CanonAdapter(BasePlatformAdapter):
    """Hermes gateway adapter for Canon conversations."""

    REQUIRES_EDIT_FINALIZE = True

    def __init__(self, config, **_: Any) -> None:
        super().__init__(config=config, platform=Platform("canon"))

        resolved = _resolve_canon_agent(config, raise_on_error=False)
        self.api_key = resolved.api_key
        self.profile_name = resolved.profile
        self.profile_agent_id = resolved.agent_id
        self.profile_agent_name = resolved.agent_name
        if self.profile_agent_name:
            os.environ.setdefault("HERMES_CANON_AGENT_NAME", self.profile_agent_name)

        self.base_url = (
            _config_value(config, "base_url", "CANON_BASE_URL", "")
            or resolved.base_url
            or DEFAULT_BASE_URL
        )
        self.stream_url = (
            _config_value(config, "stream_url", "CANON_STREAM_URL", "")
            or resolved.stream_url
            or DEFAULT_STREAM_URL
        )
        self.history_limit = _config_int(
            config, "history_limit", "CANON_HISTORY_LIMIT", DEFAULT_HISTORY_LIMIT
        )

        self._client: Optional[CanonHttpClient] = None
        self._agent_id: Optional[str] = None
        self._stream_task: Optional[asyncio.Task] = None
        self._runtime_status_task: Optional[asyncio.Task] = None
        self._runtime_signal_task: Optional[asyncio.Task] = None
        self._stream_stop: Optional[asyncio.Event] = None
        self._last_event_id: Optional[str] = None
        self._conversation_cache: dict[str, dict[str, Any]] = {}
        self._seen_message_ids: set[str] = set()
        self._seen_message_order: Deque[str] = deque()
        self._hitl_tasks: set[asyncio.Task] = set()
        self._runtime_turn_ids: dict[str, str] = {}
        self._runtime_turn_opened_at: dict[str, int] = {}
        self._session_conversation_ids: dict[str, str] = {}

    @property
    def name(self) -> str:
        return "Canon"

    async def connect(self) -> bool:
        if not self.api_key:
            self._set_fatal_error(
                "missing_api_key",
                "CANON_API_KEY, config.api_key, or CANON_AGENT profile is required for the Canon platform",
                retryable=False,
            )
            return False

        self._client = self._make_client()
        self._stream_stop = asyncio.Event()

        try:
            ctx = await self._client.get_me()
            self._agent_id = _first_string(ctx, "agentId", "id", "userId")
            agent_display_name = _first_string(ctx, "displayName", "agentName", "name")
            if agent_display_name:
                self.profile_agent_name = agent_display_name
                os.environ["HERMES_CANON_AGENT_NAME"] = agent_display_name
            await self._refresh_conversations()
            self._mark_connected()
            await self._publish_runtime_status()
            self._runtime_status_task = asyncio.create_task(
                self._runtime_status_loop(), name="canon-platform-runtime-status"
            )
            self._runtime_signal_task = asyncio.create_task(
                self._runtime_signal_loop(), name="canon-platform-runtime-signals"
            )
            self._stream_task = asyncio.create_task(
                self._stream_loop(), name="canon-platform-stream"
            )
            logger.info(
                "Canon platform connected as agent %s", self._agent_id or "<unknown>"
            )
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._set_fatal_error(
                "connect_failed", _safe_error(exc), retryable=_is_retryable(exc)
            )
            await self._close_client()
            return False

    async def disconnect(self) -> None:
        stop = self._stream_stop
        if stop is not None:
            stop.set()

        task = self._stream_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug(
                    "Canon stream task raised during disconnect", exc_info=True
                )

        status_task = self._runtime_status_task
        if status_task and not status_task.done():
            status_task.cancel()
            try:
                await status_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug(
                    "Canon runtime status task raised during disconnect", exc_info=True
                )

        signal_task = self._runtime_signal_task
        if signal_task and not signal_task.done():
            signal_task.cancel()
            try:
                await signal_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug(
                    "Canon runtime signal task raised during disconnect", exc_info=True
                )

        if self._hitl_tasks:
            tasks = list(self._hitl_tasks)
            for hitl_task in tasks:
                hitl_task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._hitl_tasks.clear()

        self._stream_task = None
        self._runtime_status_task = None
        self._runtime_signal_task = None
        self._stream_stop = None
        await self._close_client()
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self.api_key:
            return SendResult(
                success=False, error="Canon API key is not configured", retryable=False
            )

        client = self._client or self._make_client()
        owns_client = self._client is None

        try:
            if _is_canon_streaming_preview(metadata):
                turn_id = self._turn_id_for_conversation(chat_id)
                await client.set_streaming(
                    chat_id,
                    text=content,
                    status="streaming",
                    turn_id=turn_id,
                )
                await self._publish_runtime_turn(
                    chat_id,
                    "streaming",
                    session_key=self._session_key_for_conversation(chat_id),
                )
                return SendResult(
                    success=True,
                    message_id=turn_id,
                    raw_response={"streaming": True, "turnId": turn_id},
                )

            canon_options, message_metadata = _split_canon_metadata(metadata)
            message_metadata.update(TURN_COMPLETE_METADATA)
            turn_id = self._active_turn_id_for_conversation(chat_id)
            if turn_id:
                message_metadata.setdefault("turnId", turn_id)
            data = await client.send_message(
                chat_id,
                content,
                reply_to=reply_to,
                metadata=message_metadata,
                options=canon_options,
            )
            message_id = _first_string(data, "messageId", "id")
            return SendResult(
                success=True,
                message_id=message_id,
                raw_response=data,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return SendResult(
                success=False,
                error=_safe_error(exc),
                retryable=_is_retryable(exc),
            )
        finally:
            if owns_client:
                await client.close()

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self.api_key:
            return SendResult(
                success=False, error="Canon API key is not configured", retryable=False
            )

        client = self._client or self._make_client()
        owns_client = self._client is None
        turn_id = message_id or self._turn_id_for_conversation(chat_id)

        try:
            if finalize:
                canon_options, message_metadata = _split_canon_metadata(metadata)
                message_metadata.update(TURN_COMPLETE_METADATA)
                message_metadata.setdefault("turnId", turn_id)
                data = await client.send_message(
                    chat_id,
                    content,
                    metadata=message_metadata,
                    options=canon_options,
                )
                try:
                    await asyncio.sleep(FINAL_MESSAGE_HANDOFF_SECONDS)
                    await client.clear_streaming(chat_id)
                except Exception:
                    logger.debug("Canon streaming clear after finalize failed", exc_info=True)
                return SendResult(
                    success=True,
                    message_id=_first_string(data, "messageId", "id"),
                    raw_response=data,
                )

            await client.set_streaming(
                chat_id,
                text=content,
                status="streaming",
                turn_id=turn_id,
            )
            await self._publish_runtime_turn(
                chat_id,
                "streaming",
                session_key=self._session_key_for_conversation(chat_id),
            )
            return SendResult(
                success=True,
                message_id=turn_id,
                raw_response={"streaming": True, "turnId": turn_id},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return SendResult(
                success=False,
                error=_safe_error(exc),
                retryable=_is_retryable(exc),
            )
        finally:
            if owns_client:
                await client.close()

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await self._send_attachment(
            chat_id,
            caption or "",
            content_type="image",
            attachment={"kind": "image", "url": image_url},
            reply_to=reply_to,
            metadata=metadata,
        )

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_media_file(
            chat_id,
            image_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
            content_type="image",
            mime_hint=kwargs.get("mime_type"),
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_media_file(
            chat_id,
            audio_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
            content_type="audio",
            mime_hint=kwargs.get("mime_type"),
            duration_ms=kwargs.get("duration_ms"),
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_media_file(
            chat_id,
            video_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
            content_type="file",
            mime_hint=kwargs.get("mime_type"),
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_media_file(
            chat_id,
            file_path,
            caption=caption,
            file_name=file_name,
            reply_to=reply_to,
            metadata=metadata,
            content_type="file",
            mime_hint=kwargs.get("mime_type"),
        )

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        if not self.api_key:
            return
        client = self._client or self._make_client()
        owns_client = self._client is None
        try:
            status = "thinking"
            if isinstance(metadata, dict) and metadata.get("status") in {
                "typing",
                "thinking",
            }:
                status = metadata["status"]
            await client.set_typing(chat_id, True, status)
        except Exception:
            logger.debug("Canon typing indicator failed", exc_info=True)
        finally:
            if owns_client:
                await client.close()

    async def stop_typing(self, chat_id: str) -> None:
        if not self.api_key:
            return
        client = self._client or self._make_client()
        owns_client = self._client is None
        try:
            await client.set_typing(chat_id, False)
        except Exception:
            logger.debug("Canon typing clear failed", exc_info=True)
        finally:
            if owns_client:
                await client.close()

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if self._client is None:
            return SendResult(success=False, error="Canon adapter is not connected")

        timeout_seconds = _canon_timeout_seconds("HERMES_CLARIFY_TIMEOUT", 300)
        expires_at = int((time.time() + timeout_seconds) * 1000)
        canon_choices = _canon_runtime_choices(choices)

        try:
            data = await self._client.create_runtime_input_request(
                chat_id,
                input_id=clarify_id,
                kind="clarify",
                expires_at=expires_at,
                title="Clarification needed",
                prompt=question,
                choices=canon_choices,
                native={
                    "runtime": "hermes",
                    "method": "clarify",
                    "requestId": clarify_id,
                    "sessionKey": session_key,
                },
            )
            await self._publish_runtime_turn(chat_id, "waiting_input", session_key=session_key)
            task = asyncio.create_task(
                self._poll_runtime_input_response(
                    chat_id,
                    clarify_id,
                    session_key=session_key,
                    expires_at=expires_at,
                ),
                name=f"canon-runtime-input-{clarify_id}",
            )
            self._track_hitl_task(task)
            return SendResult(
                success=True,
                message_id=_first_string(data, "messageId"),
                raw_response=data,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return SendResult(
                success=False,
                error=_safe_error(exc),
                retryable=_is_retryable(exc),
            )

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if self._client is None:
            return SendResult(success=False, error="Canon adapter is not connected")

        approval_id = f"hermes-{uuid.uuid4().hex[:20]}"
        timeout_seconds = _canon_timeout_seconds("HERMES_APPROVAL_TIMEOUT", 300)
        expires_at = int((time.time() + timeout_seconds) * 1000)
        command_preview = command[:4000]
        details = [
            {"label": "Command", "value": command_preview, "monospace": True},
            {"label": "Reason", "value": description},
        ]

        try:
            data = await self._client.create_runtime_approval_request(
                chat_id,
                approval_id=approval_id,
                tool_name="Command",
                tool_summary=description or "Command approval required",
                expires_at=expires_at,
                details=details,
                native={
                    "runtime": "hermes",
                    "method": "exec_approval",
                    "requestId": approval_id,
                    "sessionKey": session_key,
                    "command": command_preview,
                },
                allow_session_rule=True,
            )
            await self._publish_runtime_turn(chat_id, "waiting_input", session_key=session_key)
            task = asyncio.create_task(
                self._poll_runtime_approval_response(
                    chat_id,
                    approval_id,
                    session_key=session_key,
                    expires_at=expires_at,
                ),
                name=f"canon-runtime-approval-{approval_id}",
            )
            self._track_hitl_task(task)
            return SendResult(
                success=True,
                message_id=_first_string(data, "messageId"),
                raw_response=data,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return SendResult(
                success=False,
                error=_safe_error(exc),
                retryable=_is_retryable(exc),
            )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        convo = self._conversation_cache.get(str(chat_id))
        if convo is None and self._client is not None:
            await self._refresh_conversations()
            convo = self._conversation_cache.get(str(chat_id))
        return {
            "id": str(chat_id),
            "name": _conversation_name(convo) if convo else str(chat_id),
            "type": _chat_type_from_conversation(convo),
        }

    def _make_client(self) -> CanonHttpClient:
        return CanonHttpClient(
            self.api_key,
            base_url=self.base_url,
            stream_url=self.stream_url,
        )

    async def _close_client(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.close()

    async def _publish_runtime_status(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.update_runtime_status(
                runtime="hermes",
                host_mode=True,
                runtime_descriptor=dict(CANON_HERMES_RUNTIME_DESCRIPTOR),
            )
        except Exception:
            logger.debug("Canon runtime status publish failed", exc_info=True)

    def _queue_depth_for_session(self, session_key: str) -> int:
        provider = getattr(self, "_queue_depth_provider", None)
        if callable(provider):
            try:
                return max(0, int(provider(session_key)))
            except Exception:
                logger.debug("Canon queue depth provider failed", exc_info=True)
        return 1 if session_key in self._pending_messages else 0

    def _turn_id_for_session(self, session_key: str) -> tuple[str, int]:
        turn_id = self._runtime_turn_ids.get(session_key)
        if not turn_id:
            turn_id = f"hermes-{uuid.uuid4().hex[:20]}"
            self._runtime_turn_ids[session_key] = turn_id
            self._runtime_turn_opened_at[session_key] = int(time.time() * 1000)
        return turn_id, self._runtime_turn_opened_at[session_key]

    def _turn_id_for_conversation(self, conversation_id: str) -> str:
        if conversation_id in self._runtime_turn_ids:
            return self._runtime_turn_ids[conversation_id]
        for session_key, known_conversation_id in self._session_conversation_ids.items():
            if known_conversation_id == conversation_id:
                return self._turn_id_for_session(session_key)[0]
        return self._turn_id_for_session(conversation_id)[0]

    def _active_turn_id_for_conversation(self, conversation_id: str) -> Optional[str]:
        if conversation_id in self._runtime_turn_ids:
            return self._runtime_turn_ids[conversation_id]
        for session_key, known_conversation_id in self._session_conversation_ids.items():
            if known_conversation_id == conversation_id:
                return self._runtime_turn_ids.get(session_key)
        return None

    def _session_key_for_conversation(self, conversation_id: str) -> str:
        for session_key, known_conversation_id in self._session_conversation_ids.items():
            if known_conversation_id == conversation_id:
                return session_key
        return conversation_id

    async def _publish_runtime_turn(
        self,
        conversation_id: str,
        state: str,
        *,
        session_key: Optional[str] = None,
        active_message_ids: Optional[list[str]] = None,
    ) -> None:
        if self._client is None or not conversation_id:
            return
        session_key = session_key or conversation_id
        self._session_conversation_ids[session_key] = conversation_id
        queue_depth = self._queue_depth_for_session(session_key)
        open_state = state in {"thinking", "streaming", "tool", "waiting_input"}
        turn_id: Optional[str] = None
        opened_at: Optional[int] = None
        if open_state:
            turn_id, opened_at = self._turn_id_for_session(session_key)
        else:
            turn_id = self._runtime_turn_ids.pop(session_key, None)
            opened_at = self._runtime_turn_opened_at.pop(session_key, None)
        capabilities = {
            "supportsInterrupt": True,
            "supportsQueue": True,
            "supportsInputInterrupt": state != "waiting_input",
            "supportsNonFinalPermanentMessages": False,
        }
        try:
            await self._client.update_runtime_turn(
                conversation_id,
                state=state,
                turn_id=turn_id,
                queue_depth=queue_depth,
                active_message_ids=active_message_ids,
                capabilities=capabilities,
                opened_at=opened_at,
                turn_updated_at=int(time.time() * 1000),
            )
            if not open_state and queue_depth <= 0:
                self._session_conversation_ids.pop(session_key, None)
        except Exception:
            logger.debug("Canon runtime turn publish failed", exc_info=True)

    async def _on_runtime_turn_start(self, event: MessageEvent, session_key: str) -> None:
        conversation_id = str(event.source.chat_id)
        self._session_conversation_ids[session_key] = conversation_id
        await self._publish_runtime_turn(
            conversation_id,
            "thinking",
            session_key=session_key,
            active_message_ids=[event.message_id] if event.message_id else None,
        )

    async def on_gateway_message_accepted(
        self,
        event: MessageEvent,
        session_key: str,
        *,
        phase: str = "active",
    ) -> None:
        raw_message = getattr(event, "raw_message", None)
        if not isinstance(raw_message, dict) or not isinstance(raw_message.get("message"), dict):
            return
        conversation_id = str(event.source.chat_id)
        self._session_conversation_ids[session_key] = conversation_id
        if self._client is None:
            return

        metadata = _canon_message_metadata(raw_message)
        if metadata.get("inboundDisposition") == "queued" and event.message_id:
            try:
                await self._client.update_message_disposition(
                    conversation_id,
                    event.message_id,
                    "accepted_now",
                )
            except Exception:
                logger.debug("Canon queued disposition update failed", exc_info=True)

        try:
            await self._client.mark_as_read(conversation_id)
        except Exception:
            logger.debug("Canon mark-as-read failed", exc_info=True)

    async def _on_runtime_turn_complete(
        self,
        event: MessageEvent,
        session_key: str,
        outcome: Any,
    ) -> None:
        conversation_id = self._session_conversation_ids.get(session_key) or str(event.source.chat_id)
        state = "interrupted" if getattr(outcome, "value", outcome) == "cancelled" else "completed"
        await self._publish_runtime_turn(conversation_id, state, session_key=session_key)

    async def _on_runtime_queue_changed(self, session_key: str) -> None:
        conversation_id = self._session_conversation_ids.get(session_key)
        if not conversation_id or session_key in self._runtime_turn_ids:
            if conversation_id:
                await self._publish_runtime_turn(conversation_id, "thinking", session_key=session_key)
            return
        await self._publish_runtime_turn(conversation_id, "idle", session_key=session_key)

    async def _runtime_status_loop(self) -> None:
        while self._stream_stop is not None and not self._stream_stop.is_set():
            try:
                await asyncio.sleep(RUNTIME_STATUS_INTERVAL_SECONDS)
                await self._publish_runtime_status()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Canon runtime status loop failed", exc_info=True)

    async def _runtime_signal_loop(self) -> None:
        while self._stream_stop is not None and not self._stream_stop.is_set():
            try:
                await self._poll_runtime_signals_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Canon runtime signal poll failed", exc_info=True)
            await asyncio.sleep(RUNTIME_SIGNAL_POLL_SECONDS)

    async def _poll_runtime_signals_once(self) -> None:
        if self._client is None:
            return
        for conversation_id in list(self._conversation_cache.keys()):
            try:
                response = await self._client.consume_runtime_signal(conversation_id)
            except Exception:
                logger.debug("Canon runtime signal consume failed", exc_info=True)
                continue
            if response.get("status") != "signal":
                continue
            signal = str(response.get("signal") or "")
            if signal in {"interrupt", "stop_and_drop", "new_session"}:
                await self._handle_runtime_signal(
                    conversation_id,
                    signal,
                    updated_by=_first_string(response, "updatedBy"),
                )

    async def _handle_runtime_signal(
        self,
        conversation_id: str,
        signal: str,
        *,
        updated_by: Optional[str] = None,
    ) -> None:
        conversation = self._conversation_cache.get(conversation_id)
        source = self.build_source(
            chat_id=conversation_id,
            chat_name=_conversation_name(conversation),
            chat_type=_chat_type_from_conversation(conversation),
            user_id=updated_by or self._agent_id or "canon-runtime-control",
            user_name="Canon",
        )
        session_key = build_session_key(
            source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
        )
        if signal == "stop_and_drop":
            self._pending_messages.pop(session_key, None)

        event = MessageEvent(
            text="",
            message_type=MessageType.TEXT,
            source=source,
            raw_message={"canonSignal": signal},
            message_id=f"canon-control:{signal}:{time.time()}",
        )
        handler = getattr(self, "_runtime_control_handler", None)
        if callable(handler):
            handled = await handler(event, session_key, signal)
            if handled:
                await self._publish_runtime_turn(
                    conversation_id,
                    "idle" if signal == "new_session" else "interrupted",
                    session_key=session_key,
                )
                return

        # Backward-compatible fallback for older Hermes gateway runners that do
        # not expose the runtime-control bridge yet.
        event.text = "/new" if signal == "new_session" else "/stop"
        await self.handle_message(event)

        if signal == "stop_and_drop":
            self._pending_messages.pop(session_key, None)

    async def _poll_runtime_input_response(
        self,
        conversation_id: str,
        input_id: str,
        *,
        session_key: str,
        expires_at: int,
    ) -> None:
        if self._client is None:
            return
        while time.time() * 1000 <= expires_at:
            try:
                response = await self._client.consume_runtime_input_response(
                    conversation_id,
                    input_id,
                )
            except Exception:
                logger.debug("Canon runtime input consume failed", exc_info=True)
                await asyncio.sleep(RUNTIME_INPUT_POLL_SECONDS)
                continue
            status = response.get("status")
            if status == "submitted":
                await self._publish_runtime_turn(conversation_id, "tool", session_key=session_key)
                value = _runtime_input_response_value(response)
                from tools.clarify_gateway import resolve_gateway_clarify

                resolve_gateway_clarify(input_id, value)
                return
            if status == "cancelled":
                await self._publish_runtime_turn(conversation_id, "interrupted", session_key=session_key)
                from tools.clarify_gateway import resolve_gateway_clarify

                resolve_gateway_clarify(input_id, "[user cancelled the clarification]")
                return
            if status == "timeout":
                await self._publish_runtime_turn(conversation_id, "interrupted", session_key=session_key)
                from tools.clarify_gateway import resolve_gateway_clarify

                resolve_gateway_clarify(input_id, "[user did not respond within 5m]")
                return
            await asyncio.sleep(RUNTIME_INPUT_POLL_SECONDS)

        try:
            await self._client.consume_runtime_input_response(
                conversation_id,
                input_id,
                cancel=True,
            )
        finally:
            await self._publish_runtime_turn(conversation_id, "interrupted", session_key=session_key)
            from tools.clarify_gateway import resolve_gateway_clarify

            resolve_gateway_clarify(input_id, "[user did not respond within 5m]")

    async def _poll_runtime_approval_response(
        self,
        conversation_id: str,
        approval_id: str,
        *,
        session_key: str,
        expires_at: int,
    ) -> None:
        if self._client is None:
            return
        while time.time() * 1000 <= expires_at:
            try:
                response = await self._client.consume_runtime_approval_response(
                    conversation_id,
                    approval_id,
                )
            except Exception:
                logger.debug("Canon runtime approval consume failed", exc_info=True)
                await asyncio.sleep(RUNTIME_APPROVAL_POLL_SECONDS)
                continue
            status = response.get("status")
            if status == "allow":
                await self._publish_runtime_turn(conversation_id, "tool", session_key=session_key)
                choice = _approval_choice_from_response(response)
                from tools.approval import resolve_gateway_approval

                resolve_gateway_approval(session_key, choice)
                return
            if status in {"deny", "timeout"}:
                await self._publish_runtime_turn(conversation_id, "interrupted", session_key=session_key)
                from tools.approval import resolve_gateway_approval

                resolve_gateway_approval(session_key, "deny")
                return
            await asyncio.sleep(RUNTIME_APPROVAL_POLL_SECONDS)

        try:
            await self._client.consume_runtime_approval_response(
                conversation_id,
                approval_id,
                cancel=True,
            )
        finally:
            await self._publish_runtime_turn(conversation_id, "interrupted", session_key=session_key)
            from tools.approval import resolve_gateway_approval

            resolve_gateway_approval(session_key, "deny")

    def _track_hitl_task(self, task: asyncio.Task) -> None:
        self._hitl_tasks.add(task)

        def _done(done_task: asyncio.Task) -> None:
            self._hitl_tasks.discard(done_task)
            try:
                done_task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("Canon HITL poll task failed", exc_info=True)

        task.add_done_callback(_done)

    async def _refresh_conversations(self) -> None:
        if self._client is None:
            return
        conversations = await self._client.get_conversations()
        for convo in conversations:
            convo_id = _first_string(convo, "id", "conversationId")
            if convo_id:
                self._conversation_cache[convo_id] = convo

    async def _send_media_file(
        self,
        chat_id: str,
        file_path: str,
        *,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        content_type: str,
        mime_hint: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> SendResult:
        if not self.api_key:
            return SendResult(
                success=False, error="Canon API key is not configured", retryable=False
            )

        path = Path(file_path).expanduser()
        if not path.exists():
            return SendResult(success=False, error=f"Media file not found: {file_path}")
        if path.stat().st_size > MAX_MEDIA_BYTES:
            return SendResult(success=False, error="Canon media upload limit is 10MB")

        client = self._client or self._make_client()
        owns_client = self._client is None
        try:
            media_bytes = path.read_bytes()
            mime_type = _guess_mime_type(str(path), mime_hint)
            uploaded = await client.upload_media(
                chat_id,
                base64.b64encode(media_bytes).decode("ascii"),
                mime_type,
                file_name=file_name or path.name,
            )
            attachment = dict(uploaded.get("attachment") or {})
            if not attachment:
                attachment = {
                    "kind": _canon_attachment_kind_for_mime(mime_type),
                    "url": uploaded.get("url"),
                    "mimeType": mime_type,
                    "fileName": file_name or path.name,
                    "sizeBytes": len(media_bytes),
                }
            if duration_ms and attachment.get("kind") == "audio":
                attachment["durationMs"] = int(duration_ms)
            if mime_type.startswith("video/"):
                content_type = "file"
            elif attachment.get("kind") in {"image", "audio"}:
                content_type = str(attachment["kind"])

            return await self._send_attachment(
                chat_id,
                caption or "",
                content_type=content_type,
                attachment=attachment,
                reply_to=reply_to,
                metadata=metadata,
                client=client,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return SendResult(
                success=False, error=_safe_error(exc), retryable=_is_retryable(exc)
            )
        finally:
            if owns_client:
                await client.close()

    async def _send_attachment(
        self,
        chat_id: str,
        text: str,
        *,
        content_type: str,
        attachment: dict[str, Any],
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        client: Optional[CanonHttpClient] = None,
    ) -> SendResult:
        if not self.api_key:
            return SendResult(
                success=False, error="Canon API key is not configured", retryable=False
            )

        active_client = client or self._client or self._make_client()
        owns_client = client is None and self._client is None
        try:
            canon_options, message_metadata = _split_canon_metadata(metadata)
            message_metadata.update(TURN_COMPLETE_METADATA)
            turn_id = self._active_turn_id_for_conversation(chat_id)
            if turn_id:
                message_metadata.setdefault("turnId", turn_id)
            canon_options.update({
                "contentType": content_type,
                "attachments": [attachment],
            })
            data = await active_client.send_message(
                chat_id,
                text,
                reply_to=reply_to,
                metadata=message_metadata,
                options=canon_options,
            )
            return SendResult(
                success=True,
                message_id=_first_string(data, "messageId", "id"),
                raw_response=data,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return SendResult(
                success=False, error=_safe_error(exc), retryable=_is_retryable(exc)
            )
        finally:
            if owns_client:
                await active_client.close()

    async def _stream_loop(self) -> None:
        assert self._client is not None
        backoff = 1.0

        while self._stream_stop is not None and not self._stream_stop.is_set():
            try:
                async for frame in self._client.stream_events(
                    last_event_id=self._last_event_id
                ):
                    if frame.event_id:
                        self._last_event_id = frame.event_id
                    await self._handle_stream_frame(frame)
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stream_stop is not None and self._stream_stop.is_set():
                    break
                logger.warning("Canon stream disconnected: %s", _safe_error(exc))
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _handle_stream_frame(self, frame: CanonStreamFrame) -> None:
        if frame.event == "agent.context" and isinstance(frame.data, dict):
            self._agent_id = (
                _first_string(frame.data, "agentId", "id", "userId") or self._agent_id
            )
            return
        if frame.event == "conversation.updated" and isinstance(frame.data, dict):
            convo_id = _first_string(frame.data, "conversationId", "id")
            if convo_id:
                cached = self._conversation_cache.setdefault(convo_id, {"id": convo_id})
                changes = frame.data.get("changes")
                if isinstance(changes, dict):
                    cached.update(changes)
            return
        if frame.event == "message.deleted" and isinstance(frame.data, dict):
            conversation_id = _first_string(frame.data, "conversationId")
            message_id = _first_string(frame.data, "messageId", "id")
            handler = getattr(self, "_queued_message_delete_handler", None)
            if conversation_id and message_id and callable(handler):
                removed = handler(conversation_id, message_id)
                if removed:
                    logger.debug(
                        "Removed %d queued Canon message(s) after deletion: %s",
                        removed,
                        message_id,
                    )
                    for session_key, mapped_conversation_id in list(self._session_conversation_ids.items()):
                        if mapped_conversation_id == conversation_id:
                            await self._on_runtime_queue_changed(session_key)
            return
        if frame.event != "message.created":
            return
        if isinstance(frame.data, dict):
            await self._handle_message_payload(frame.data)

    async def _handle_message_payload(self, payload: dict[str, Any]) -> None:
        message = payload.get("message")
        if not isinstance(message, dict):
            return

        sender_id = _first_string(message, "senderId")
        if self._agent_id and sender_id == self._agent_id:
            return
        if _is_canon_control_message(message):
            return

        message_id = _first_string(message, "id")
        if message_id and self._already_seen(message_id):
            return

        conversation_id = _first_string(payload, "conversationId")
        if not conversation_id:
            return

        conversation = payload.get("conversation")
        if isinstance(conversation, dict):
            self._conversation_cache[conversation_id] = conversation
        conversation = self._conversation_cache.get(
            conversation_id, conversation if isinstance(conversation, dict) else None
        )

        text = _message_text(message)
        if not text:
            return

        (
            media_urls,
            media_types,
            media_message_type,
        ) = await self._materialize_attachments(message)

        source = self.build_source(
            chat_id=conversation_id,
            chat_name=_conversation_name(conversation),
            chat_type=_chat_type_from_conversation(conversation),
            user_id=sender_id,
            user_name=_first_string(message, "senderName"),
            message_id=message_id,
        )

        event = MessageEvent(
            text=text,
            message_type=_message_type_for_text_and_media(text, media_message_type),
            source=source,
            raw_message=payload,
            message_id=message_id,
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=_first_string(message, "replyTo"),
            channel_prompt=_canon_channel_prompt(
                payload,
                conversation,
                self._agent_id,
                self.profile_agent_name,
            ),
        )
        await self.handle_message(event)

    async def _materialize_attachments(
        self,
        message: dict[str, Any],
    ) -> tuple[list[str], list[str], Optional[MessageType]]:
        attachments = message.get("attachments")
        if not isinstance(attachments, list) or not attachments:
            return [], [], None
        if self._client is None:
            return [], [], None

        media_urls: list[str] = []
        media_types: list[str] = []
        message_type: Optional[MessageType] = None

        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            url = _first_string(attachment, "url")
            if not url:
                continue

            try:
                data, response_mime = await self._client.download_media(url)
                mime_type = _attachment_mime_type(attachment, response_mime)
                local_path, local_message_type = _cache_canon_media(
                    attachment, data, mime_type
                )
            except Exception:
                logger.debug(
                    "Failed to materialize Canon media attachment", exc_info=True
                )
                continue

            media_urls.append(local_path)
            media_types.append(mime_type)
            message_type = _prefer_media_message_type(message_type, local_message_type)

        return media_urls, media_types, message_type

    def _already_seen(self, message_id: str) -> bool:
        if message_id in self._seen_message_ids:
            return True
        self._seen_message_ids.add(message_id)
        self._seen_message_order.append(message_id)
        while len(self._seen_message_order) > MAX_SEEN_MESSAGE_IDS:
            old = self._seen_message_order.popleft()
            self._seen_message_ids.discard(old)
        return False


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


def _config_extra_value(config: Any, *keys: str) -> str:
    extra = getattr(config, "extra", {}) or {}
    if not isinstance(extra, dict):
        return ""
    for key in keys:
        value = extra.get(key)
        if value:
            return str(value).strip()
    return ""


def _canon_home() -> Path:
    configured = os.getenv("CANON_HOME", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".canon"


def _canon_agents_path() -> Path:
    configured = (
        os.getenv("CANON_AGENTS_FILE", "").strip()
        or os.getenv("CANON_AGENTS_PATH", "").strip()
    )
    if configured:
        return Path(configured).expanduser()
    return _canon_home() / "agents.json"


def _parse_agent_profiles(raw: str) -> dict[str, dict[str, Any]]:
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(base64.b64decode(text).decode("utf-8"))
        except Exception as exc:
            raise ValueError(
                f"{CANON_AGENTS_JSON_BOOTSTRAP_ENV} must be agents.json JSON or base64 JSON"
            ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"{CANON_AGENTS_JSON_BOOTSTRAP_ENV} must be a JSON object keyed by profile name"
        )
    profiles: dict[str, dict[str, Any]] = {}
    for raw_name, value in parsed.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"{CANON_AGENTS_JSON_BOOTSTRAP_ENV} contains an empty profile name")
        if not isinstance(value, dict):
            raise ValueError(
                f'{CANON_AGENTS_JSON_BOOTSTRAP_ENV} profile "{name}" must be an object'
            )
        profiles[name] = value
    return profiles


def _write_agent_profiles(path: Path, profiles: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(profiles, indent=2, sort_keys=True), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def _load_agent_profiles() -> dict[str, dict[str, Any]]:
    path = _canon_agents_path()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        loaded = {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a JSON object keyed by profile name")

    profiles = {
        str(name): value
        for name, value in loaded.items()
        if isinstance(value, dict)
    }

    bootstrap = os.getenv(CANON_AGENTS_JSON_BOOTSTRAP_ENV, "")
    if bootstrap:
        changed = False
        for name, profile in _parse_agent_profiles(bootstrap).items():
            if name not in profiles:
                profiles[name] = profile
                changed = True
        if changed or not path.exists():
            _write_agent_profiles(path, profiles)

    return profiles


def _profile_slug(name: str) -> str:
    slug_chars: list[str] = []
    prev_dash = False
    for char in name.strip().lower():
        if char.isascii() and char.isalnum():
            slug_chars.append(char)
            prev_dash = False
        elif not prev_dash:
            slug_chars.append("-")
            prev_dash = True
    slug = "".join(slug_chars).strip("-")
    return slug or "hermes"


def _profile_matches_client(profile: dict[str, Any], expected: str = "hermes") -> bool:
    client_type = profile.get("clientType")
    return not client_type or str(client_type) == expected


def _resolved_from_profile(name: str, profile: dict[str, Any]) -> CanonResolvedAgent:
    if not _profile_matches_client(profile):
        raise ValueError(f'Canon profile "{name}" is not registered for Hermes')
    api_key = str(profile.get("apiKey") or "").strip()
    if not api_key:
        raise ValueError(f'Canon profile "{name}" is missing apiKey')
    return CanonResolvedAgent(
        api_key=api_key,
        profile=name,
        agent_id=str(profile.get("agentId") or "").strip() or None,
        agent_name=str(profile.get("agentName") or "").strip() or None,
        base_url=str(profile.get("baseUrl") or "").strip(),
        stream_url=str(profile.get("streamUrl") or "").strip(),
    )


def _resolve_canon_agent(config: Any, *, raise_on_error: bool = True) -> CanonResolvedAgent:
    api_key = _config_value(config, "api_key", "CANON_API_KEY", "")
    if api_key:
        return CanonResolvedAgent(api_key=api_key)

    profile_name = (
        os.getenv("CANON_AGENT", "").strip()
        or _config_extra_value(config, "agent", "profile", "canon_agent", "CANON_AGENT")
    )
    try:
        profiles = _load_agent_profiles()
        if profile_name:
            profile = profiles.get(profile_name)
            if not profile:
                raise ValueError(
                    f'Canon profile "{profile_name}" not found in {_canon_agents_path()}. '
                    f"Set {CANON_AGENTS_JSON_BOOTSTRAP_ENV}, set CANON_API_KEY, or re-run registration."
                )
            return _resolved_from_profile(profile_name, profile)

        viable = [
            (name, profile)
            for name, profile in profiles.items()
            if _profile_matches_client(profile)
        ]
        if len(viable) == 1:
            return _resolved_from_profile(viable[0][0], viable[0][1])
        if len(viable) > 1:
            raise ValueError(
                "Multiple Canon profiles are available. Set CANON_AGENT to choose one."
            )
    except Exception:
        if raise_on_error:
            raise
        return CanonResolvedAgent()
    return CanonResolvedAgent()


def _config_value(config: Any, key: str, env_name: str, default: str = "") -> str:
    env_value = os.getenv(env_name)
    if env_value:
        return env_value.strip()

    if key == "api_key":
        for attr in ("api_key", "token"):
            value = getattr(config, attr, None)
            if value:
                return str(value).strip()

    extra = getattr(config, "extra", {}) or {}
    if isinstance(extra, dict):
        for candidate in (key, env_name.lower(), env_name):
            value = extra.get(candidate)
            if value:
                return str(value).strip()
    return default


def _config_int(config: Any, key: str, env_name: str, default: int) -> int:
    raw = _config_value(config, key, env_name, "")
    if raw == "":
        return default
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return default


def _save_canon_profile(
    profile_name: str,
    *,
    api_key: str,
    agent_id: str,
    agent_name: str,
    base_url: str = "",
    stream_url: str = "",
) -> None:
    profiles = _load_agent_profiles()
    entry: dict[str, Any] = {
        "apiKey": api_key,
        "agentId": agent_id,
        "agentName": agent_name,
        "registeredAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clientType": "hermes",
    }
    if base_url and base_url.rstrip("/") != DEFAULT_BASE_URL:
        entry["baseUrl"] = base_url.rstrip("/")
    if stream_url and stream_url.rstrip("/") != DEFAULT_STREAM_URL:
        entry["streamUrl"] = stream_url.rstrip("/")
    profiles[profile_name] = entry
    _write_agent_profiles(_canon_agents_path(), profiles)


def _post_registration_request(
    *,
    base_url: str,
    name: str,
    description: str,
    owner_phone: str,
    requested_agent_id: Optional[str] = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "description": description,
        "ownerPhone": owner_phone,
        "developerInfo": "Hermes gateway platform adapter",
        "clientType": "hermes",
        "localRegistrationId": str(uuid.uuid4()),
    }
    if requested_agent_id:
        body["requestedAgentId"] = requested_agent_id

    res = httpx.post(
        f"{base_url.rstrip('/')}/agents/register",
        headers={"Content-Type": "application/json"},
        json=body,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    if res.status_code >= 400:
        raise CanonApiError(res.status_code, res.text)
    data = res.json()
    if not isinstance(data, dict) or not data.get("requestId"):
        raise ValueError("Canon registration did not return a requestId")
    return data


def _get_registration_status(
    *,
    base_url: str,
    request_id: str,
    poll_token: Optional[str],
) -> dict[str, Any]:
    headers = {"x-canon-poll-token": poll_token} if poll_token else None
    res = httpx.get(
        f"{base_url.rstrip('/')}/agents/status/{request_id}",
        headers=headers,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    if res.status_code >= 400:
        raise CanonApiError(res.status_code, res.text)
    data = res.json()
    if not isinstance(data, dict):
        raise ValueError("Canon registration status response was not an object")
    return data


def _ack_registration_status(
    *,
    base_url: str,
    request_id: str,
    poll_token: Optional[str],
) -> None:
    headers = {"x-canon-poll-token": poll_token} if poll_token else None
    res = httpx.post(
        f"{base_url.rstrip('/')}/agents/status/{request_id}/ack",
        headers=headers,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    if res.status_code >= 400:
        raise CanonApiError(res.status_code, res.text)


def _wait_for_registration_approval(
    *,
    base_url: str,
    request_id: str,
    poll_token: Optional[str],
    on_poll: Optional[Callable[[dict[str, Any]], None]] = None,
    timeout_seconds: float = REGISTRATION_TIMEOUT_SECONDS,
    poll_interval_seconds: float = REGISTRATION_POLL_INTERVAL_SECONDS,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        time.sleep(poll_interval_seconds)
        status = _get_registration_status(
            base_url=base_url,
            request_id=request_id,
            poll_token=poll_token,
        )
        if on_poll:
            on_poll(status)
        if status.get("status") in {"approved", "rejected"}:
            return status
    return {"status": "timeout"}


def _setup_canon() -> None:
    from hermes_cli.config import get_env_value, remove_env_value, save_env_value
    from hermes_cli.setup import (
        Colors,
        color,
        print_error,
        print_info,
        print_success,
        print_warning,
        prompt,
        prompt_choice,
        prompt_yes_no,
    )

    print()
    print(color("  --- Canon Setup ---", Colors.CYAN))
    print()
    print_info("  Canon can use an existing agent profile or register a new Hermes agent.")
    print_info("  Registration sends an approval request to the Canon owner app.")

    existing_key = get_env_value("CANON_API_KEY") or ""
    existing_agent = get_env_value("CANON_AGENT") or ""
    if existing_key or existing_agent or validate_config(None):
        print()
        if existing_agent:
            print_success(f"Canon is already configured with profile '{existing_agent}'.")
        else:
            print_success("Canon is already configured.")
        if not prompt_yes_no("  Reconfigure Canon?", False):
            return

    choices = [
        "Register/reconnect a Hermes agent with Canon (recommended)",
        "Use an existing Canon profile from ~/.canon/agents.json",
        "Paste an existing Canon API key",
    ]
    choice = prompt_choice("  How would you like to configure Canon?", choices, 0)

    if choice == 2:
        print()
        print_info("  Paste the agk_live_... key returned by Canon registration.")
        api_key = prompt("  Canon API key", password=True)
        if not api_key:
            print_warning("  Skipped — Canon needs CANON_API_KEY or CANON_AGENT.")
            return
        save_env_value("CANON_API_KEY", api_key)
        remove_env_value("CANON_AGENT")
        print_success("  Canon API key saved.")
        return

    if choice == 1:
        try:
            profiles = _load_agent_profiles()
        except Exception as exc:
            print_error(f"  Could not read {_canon_agents_path()}: {_safe_error(exc)}")
            return
        available = [
            name
            for name, profile in sorted(profiles.items())
            if isinstance(profile, dict) and _profile_matches_client(profile)
        ]
        if available:
            default = existing_agent if existing_agent in available else available[0]
            profile_name = prompt("  Canon profile", default=default)
        else:
            print_warning("  No Hermes-compatible profiles found in ~/.canon/agents.json.")
            profile_name = prompt("  Canon profile")
        if not profile_name:
            print_warning("  Skipped — Canon needs a profile name.")
            return
        try:
            _resolved_from_profile(profile_name, profiles[profile_name])
        except KeyError:
            print_error(f"  Profile '{profile_name}' was not found in {_canon_agents_path()}.")
            return
        except Exception as exc:
            print_error(f"  Profile '{profile_name}' is not usable: {_safe_error(exc)}")
            return
        save_env_value("CANON_AGENT", profile_name)
        remove_env_value("CANON_API_KEY")
        print_success(f"  Canon profile '{profile_name}' saved.")
        return

    print()
    agent_name = prompt("  Agent display name", default="Hermes")
    if not agent_name:
        print_warning("  Skipped — Canon registration needs an agent name.")
        return
    description = prompt(
        "  Agent description",
        default="Hermes gateway agent",
    )
    owner_phone = prompt("  Canon owner phone number (E.164, exact number from the Canon app)")
    if not owner_phone:
        print_warning("  Skipped — Canon registration needs the owner's phone number.")
        return
    profile_name = prompt("  Local Canon profile name", default=_profile_slug(agent_name))
    if not profile_name:
        print_warning("  Skipped — Canon registration needs a local profile name.")
        return

    base_url = prompt(
        "  Canon API base URL",
        default=get_env_value("CANON_BASE_URL") or DEFAULT_BASE_URL,
    ).rstrip("/")

    requested_agent_id = None
    try:
        requested_agent_id = _load_agent_profiles().get(profile_name, {}).get("agentId")
    except Exception:
        requested_agent_id = None

    try:
        submitted = _post_registration_request(
            base_url=base_url,
            name=agent_name,
            description=description,
            owner_phone=owner_phone,
            requested_agent_id=requested_agent_id,
        )
    except Exception as exc:
        print_error(f"  Canon registration failed: {_safe_error(exc)}")
        return

    request_id = str(submitted.get("requestId") or "")
    poll_token = submitted.get("pollToken")
    print_success(f"  Registration submitted (request ID: {request_id}).")
    print_info("  Approve the request in the Canon app. Waiting up to 5 minutes...")

    def _poll_update(status: dict[str, Any]) -> None:
        state = status.get("status", "pending")
        if state == "pending":
            print(".", end="", flush=True)

    try:
        result = _wait_for_registration_approval(
            base_url=base_url,
            request_id=request_id,
            poll_token=str(poll_token) if poll_token else None,
            on_poll=_poll_update,
        )
    except Exception as exc:
        print()
        print_error(f"  Canon approval polling failed: {_safe_error(exc)}")
        return

    print()
    status = result.get("status")
    if status == "rejected":
        print_warning("  Canon registration was rejected.")
        return
    if status != "approved":
        print_warning("  Canon registration timed out. Run setup again to retry.")
        return

    api_key = str(result.get("apiKey") or "").strip()
    agent_id = str(result.get("agentId") or "").strip()
    approved_name = str(result.get("agentName") or agent_name).strip()
    if not api_key or not agent_id:
        print_error("  Canon approved the request but did not return a usable API key.")
        return

    try:
        _save_canon_profile(
            profile_name,
            api_key=api_key,
            agent_id=agent_id,
            agent_name=approved_name,
            base_url=base_url,
        )
    except Exception as exc:
        print_error(f"  Could not save Canon profile: {_safe_error(exc)}")
        return

    try:
        _ack_registration_status(
            base_url=base_url,
            request_id=request_id,
            poll_token=str(poll_token) if poll_token else None,
        )
    except Exception as exc:
        print_warning(f"  Canon profile saved, but key-delivery ack failed: {_safe_error(exc)}")

    save_env_value("CANON_AGENT", profile_name)
    remove_env_value("CANON_API_KEY")
    if base_url != DEFAULT_BASE_URL:
        save_env_value("CANON_BASE_URL", base_url)

    print_success(f"  Canon agent '{approved_name}' saved as profile '{profile_name}'.")
    print_info("  Hermes will use CANON_AGENT for the Canon platform.")


def _first_string(data: Any, *keys: str) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return None


def _conversation_name(conversation: Optional[dict[str, Any]]) -> Optional[str]:
    if not isinstance(conversation, dict):
        return None
    return _first_string(conversation, "name", "topic") or _first_string(
        conversation, "id"
    )


def _chat_type_from_conversation(conversation: Optional[dict[str, Any]]) -> str:
    if isinstance(conversation, dict) and conversation.get("type") == "group":
        return "group"
    return "dm"


def _canon_channel_prompt(
    payload: dict[str, Any],
    conversation: Optional[dict[str, Any]],
    agent_id: Optional[str],
    agent_name: Optional[str],
) -> Optional[str]:
    """Surface Canon's structured group targeting metadata to the model."""
    message = payload.get("message")
    if not isinstance(message, dict) or _chat_type_from_conversation(conversation) != "group":
        return None

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        provenance = message.get("provenance") if isinstance(message.get("provenance"), dict) else {}

    mentions = message.get("mentions") or payload.get("mentions") or []
    if not isinstance(mentions, list):
        mentions = []

    mentioned_agent = bool(provenance.get("mentionedAgent"))
    if agent_id and str(agent_id) in {str(item) for item in mentions}:
        mentioned_agent = True

    name_note = f" ({agent_name})" if agent_name else ""
    if mentioned_agent:
        return (
            f"Canon group metadata: this message explicitly mentioned this agent{name_note}. "
            "Treat it as addressed to you."
        )

    return (
        f"Canon group metadata: this message did not structurally mention this agent{name_note}. "
        "Answer only if the content clearly asks for your help or your configured group policy says to participate."
    )


def _format_contact_card_text(message: dict[str, Any]) -> str:
    card = message.get("contactCard")
    if not isinstance(card, dict):
        return "[Contact card]"

    display_name = (_first_string(card, "displayName") or "Unknown").strip() or "Unknown"
    user_id = _first_string(card, "userId")
    user_type = _first_string(card, "userType")
    owner_name = _first_string(card, "ownerName")
    about = _first_string(card, "about")

    parts: list[str] = []
    if user_type:
        parts.append(user_type)
    if user_id:
        parts.append(f"userId: {user_id}")
    if owner_name:
        parts.append(f"owner: {owner_name}")
    if about:
        parts.append(f"about: {about}")

    summary = f'[Contact card] "{display_name}"'
    if parts:
        summary = f"{summary} - {'; '.join(parts)}"

    text = message.get("text")
    if isinstance(text, str) and text.strip():
        return f"{summary}\n{text.strip()}"
    return summary


def _message_text(message: dict[str, Any]) -> str:
    content_type = message.get("contentType")
    if content_type == "contact_card":
        return _format_contact_card_text(message)

    text = message.get("text")
    if isinstance(text, str) and text.strip():
        return text

    attachments = message.get("attachments")
    if isinstance(attachments, list) and attachments:
        labels: list[str] = []
        for item in attachments:
            if not isinstance(item, dict):
                continue
            kind = _first_string(item, "kind") or content_type or "file"
            name = _first_string(item, "fileName", "url")
            labels.append(
                f"{kind} attachment: {name}" if name else f"{kind} attachment"
            )
        if labels:
            return "[" + "; ".join(labels) + "]"

    if isinstance(content_type, str) and content_type != "text":
        return f"[{content_type} message]"
    return ""


def _message_type_for_text_and_media(
    text: str, media_type: Optional[MessageType]
) -> MessageType:
    if text.strip().startswith("/"):
        return MessageType.COMMAND
    return media_type or MessageType.TEXT


def _prefer_media_message_type(
    current: Optional[MessageType],
    candidate: MessageType,
) -> MessageType:
    priority = {
        MessageType.VOICE: 4,
        MessageType.AUDIO: 4,
        MessageType.VIDEO: 3,
        MessageType.PHOTO: 2,
        MessageType.DOCUMENT: 1,
    }
    if current is None:
        return candidate
    return (
        candidate if priority.get(candidate, 0) > priority.get(current, 0) else current
    )


def _guess_mime_type(path_or_name: str, override: Optional[str] = None) -> str:
    if override:
        return override
    guessed, _encoding = mimetypes.guess_type(path_or_name)
    if guessed:
        return guessed
    ext = Path(path_or_name.split("?", 1)[0]).suffix.lower()
    if ext in {".m4a"}:
        return "audio/mp4"
    if ext in {".opus"}:
        return "audio/ogg"
    if ext in VIDEO_EXTS:
        return "video/mp4"
    if ext in IMAGE_EXTS:
        return "image/jpeg"
    if ext in AUDIO_EXTS:
        return "audio/mpeg"
    return "application/octet-stream"


def _canon_attachment_kind_for_mime(mime_type: str) -> str:
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("audio/"):
        return "audio"
    return "file"


def _attachment_mime_type(
    attachment: dict[str, Any],
    response_mime: Optional[str] = None,
) -> str:
    explicit = _first_string(attachment, "mimeType")
    if explicit:
        return explicit.split(";", 1)[0].strip().lower()
    if response_mime:
        return response_mime.split(";", 1)[0].strip().lower()
    name = _first_string(attachment, "fileName", "url") or ""
    return _guess_mime_type(name)


def _attachment_file_name(attachment: dict[str, Any], mime_type: str) -> str:
    name = _first_string(attachment, "fileName")
    if name:
        return Path(name).name

    url = _first_string(attachment, "url") or ""
    url_name = Path(url.split("?", 1)[0]).name
    if url_name and "." in url_name:
        return url_name

    ext = mimetypes.guess_extension(mime_type) or ".bin"
    return f"canon-media{ext}"


def _cache_canon_media(
    attachment: dict[str, Any],
    data: bytes,
    mime_type: str,
) -> tuple[str, MessageType]:
    ext = Path(_attachment_file_name(attachment, mime_type)).suffix.lower()
    if not ext:
        ext = mimetypes.guess_extension(mime_type) or ".bin"

    if mime_type.startswith("image/"):
        return cache_image_from_bytes(
            data, ext if ext in IMAGE_EXTS else ".jpg"
        ), MessageType.PHOTO
    if mime_type.startswith("audio/"):
        return cache_audio_from_bytes(
            data, ext if ext in AUDIO_EXTS else ".ogg"
        ), MessageType.VOICE
    if mime_type.startswith("video/"):
        return cache_video_from_bytes(
            data, ext if ext in VIDEO_EXTS else ".mp4"
        ), MessageType.VIDEO
    return cache_document_from_bytes(
        data, _attachment_file_name(attachment, mime_type)
    ), MessageType.DOCUMENT


def _media_file_path(item: Any) -> str:
    if isinstance(item, (list, tuple)) and item:
        return str(item[0])
    return str(item)


def _split_canon_metadata(
    metadata: Optional[Dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(metadata, dict):
        return {}, {}

    options = metadata.get("canon_options")
    canon_metadata = metadata.get("canon_metadata")

    safe_options = dict(options) if isinstance(options, dict) else {}
    safe_metadata = dict(canon_metadata) if isinstance(canon_metadata, dict) else {}

    if not safe_metadata and metadata:
        hermes_metadata = {
            key: value
            for key, value in metadata.items()
            if key != "canon_streaming_preview"
        }
        if hermes_metadata:
            safe_metadata["hermes"] = hermes_metadata

    return safe_options, safe_metadata


def _is_canon_streaming_preview(metadata: Optional[Dict[str, Any]]) -> bool:
    return isinstance(metadata, dict) and metadata.get("canon_streaming_preview") is True


def _canon_timeout_seconds(env_name: str, default_seconds: int) -> int:
    raw = os.getenv(env_name, "").strip()
    try:
        value = int(raw) if raw else default_seconds
    except ValueError:
        value = default_seconds
    return max(5, min(value, RUNTIME_HITL_MAX_TIMEOUT_SECONDS))


def _canon_runtime_choices(choices: Optional[list]) -> Optional[list[dict[str, Any]]]:
    if not choices:
        return None
    normalized: list[dict[str, Any]] = []
    for index, choice in enumerate(choices[:12]):
        if isinstance(choice, dict):
            label = str(
                choice.get("label")
                or choice.get("text")
                or choice.get("value")
                or ""
            ).strip()
            value = str(choice.get("value") or label).strip()
        else:
            label = str(choice).strip()
            value = label
        if not label:
            continue
        normalized.append({
            "label": label[:160],
            "value": (value or label)[:400],
        })
    return normalized or None


def _runtime_input_response_value(response: dict[str, Any]) -> str:
    value = response.get("value")
    answers = response.get("answers")
    if isinstance(value, str):
        if value.strip():
            return value
        if not isinstance(answers, dict):
            return value
    if isinstance(answers, dict):
        values: list[str] = []
        for question_id, item in answers.items():
            if not isinstance(question_id, str):
                continue
            raw_answers = item.get("answers") if isinstance(item, dict) else item
            if isinstance(raw_answers, list):
                normalized = [entry.strip() for entry in raw_answers if isinstance(entry, str) and entry.strip()]
            elif isinstance(raw_answers, str) and raw_answers.strip():
                normalized = [raw_answers.strip()]
            else:
                normalized = []
            if not normalized:
                continue
            if len(answers) == 1:
                values.append(", ".join(normalized))
            else:
                values.append(f"{question_id}: {', '.join(normalized)}")
        if values:
            return "\n".join(values)
    if isinstance(value, str):
        return value
    choice = response.get("choice")
    if isinstance(choice, dict):
        for key in ("value", "label"):
            text = choice.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
    return ""


def _canon_message_metadata(raw_message: Any) -> dict[str, Any]:
    if not isinstance(raw_message, dict):
        return {}
    message = raw_message.get("message")
    if not isinstance(message, dict):
        return {}
    metadata = message.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _approval_choice_from_response(response: dict[str, Any]) -> str:
    session_rule = response.get("sessionRule")
    if isinstance(session_rule, dict):
        rule_type = session_rule.get("type")
        if rule_type in {"approve-all", "approve-tool"}:
            return "session"
    return "once"


def _is_canon_control_message(message: dict[str, Any]) -> bool:
    metadata = message.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return metadata.get("type") in CONTROL_METADATA_TYPES


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


def check_requirements() -> bool:
    """Return true when Canon credentials can be resolved."""
    return validate_config(None)


def validate_config(config) -> bool:
    try:
        return bool(_resolve_canon_agent(config).api_key)
    except Exception:
        return False


def is_connected(config) -> bool:
    return validate_config(config)


def _env_enablement() -> dict | None:
    try:
        resolved = _resolve_canon_agent(None)
    except Exception:
        return None

    if not resolved.api_key:
        return None

    seed: dict[str, Any] = {}
    if resolved.profile:
        seed["agent"] = resolved.profile
    else:
        seed["api_key"] = resolved.api_key

    if os.getenv("CANON_BASE_URL"):
        seed["base_url"] = os.getenv("CANON_BASE_URL", "").strip()
    elif resolved.base_url:
        seed["base_url"] = resolved.base_url
    if os.getenv("CANON_STREAM_URL"):
        seed["stream_url"] = os.getenv("CANON_STREAM_URL", "").strip()
    elif resolved.stream_url:
        seed["stream_url"] = resolved.stream_url
    if os.getenv("CANON_HISTORY_LIMIT"):
        seed["history_limit"] = os.getenv("CANON_HISTORY_LIMIT", "").strip()
    if os.getenv("CANON_HOME_CHANNEL"):
        seed["home_channel"] = {
            "chat_id": os.getenv("CANON_HOME_CHANNEL", "").strip(),
            "name": os.getenv("CANON_HOME_CHANNEL_NAME", "Canon Home"),
        }
    return seed


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id=None,
    media_files=None,
    force_document: bool = False,
) -> dict:
    try:
        resolved = _resolve_canon_agent(pconfig)
    except Exception as exc:
        return {"error": f"Canon standalone send failed: {_safe_error(exc)}"}

    if not resolved.api_key:
        return {
            "error": "Canon standalone send failed: Canon agent credentials are not configured"
        }

    target = chat_id or os.getenv("CANON_HOME_CHANNEL", "").strip()
    if not target:
        home = getattr(pconfig, "home_channel", None)
        target = getattr(home, "chat_id", "") or ""
    if not target:
        return {"error": "Canon standalone send failed: no conversation ID provided"}

    text = message or ""

    client = CanonHttpClient(
        resolved.api_key,
        base_url=(
            _config_value(pconfig, "base_url", "CANON_BASE_URL", "")
            or resolved.base_url
            or DEFAULT_BASE_URL
        ),
        stream_url=_config_value(
            pconfig, "stream_url", "CANON_STREAM_URL", ""
        )
        or resolved.stream_url
        or DEFAULT_STREAM_URL,
    )
    try:
        options: dict[str, Any] = {}
        if media_files:
            attachments: list[dict[str, Any]] = []
            for item in media_files:
                media_path = _media_file_path(item)
                path = Path(media_path).expanduser()
                if not path.exists():
                    return {
                        "error": f"Canon standalone send failed: media file not found: {media_path}"
                    }
                if path.stat().st_size > MAX_MEDIA_BYTES:
                    return {
                        "error": "Canon standalone send failed: Canon media upload limit is 10MB"
                    }
                mime_type = _guess_mime_type(str(path))
                uploaded = await client.upload_media(
                    target,
                    base64.b64encode(path.read_bytes()).decode("ascii"),
                    mime_type,
                    file_name=path.name,
                )
                attachment = dict(uploaded.get("attachment") or {})
                if not attachment:
                    attachment = {
                        "kind": _canon_attachment_kind_for_mime(mime_type),
                        "url": uploaded.get("url"),
                        "mimeType": mime_type,
                        "fileName": path.name,
                        "sizeBytes": path.stat().st_size,
                    }
                attachments.append(attachment)

            if attachments:
                first_kind = str(attachments[0].get("kind") or "file")
                options["contentType"] = (
                    first_kind if first_kind in {"image", "audio"} else "file"
                )
                options["attachments"] = attachments

        data = await client.send_message(
            target,
            text,
            reply_to=str(thread_id) if thread_id else None,
            metadata=dict(TURN_COMPLETE_METADATA),
            options=options or None,
        )
        return {"success": True, "message_id": _first_string(data, "messageId", "id")}
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return {"error": f"Canon standalone send failed: {_safe_error(exc)}"}
    finally:
        await client.close()


def register(ctx):
    """Plugin entry point: called by the Hermes plugin system."""
    ctx.register_platform(
        name="canon",
        label="Canon",
        adapter_factory=lambda cfg: CanonAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=[],
        install_hint="Set CANON_API_KEY or CANON_AGENT for your Canon agent",
        setup_fn=_setup_canon,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="CANON_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="CANON_ALLOWED_USERS",
        allow_all_env="CANON_ALLOW_ALL_USERS",
        group_allowed_users_env="CANON_GROUP_ALLOWED_USERS",
        group_allowed_chats_env="CANON_GROUP_ALLOWED_CONVERSATIONS",
        max_message_length=8000,
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via Canon. Canon conversations can be direct or group chats. "
            "Use concise conversational text; slash commands are preserved when users send them. "
            "Messages sent by Hermes are marked as turn-complete for Canon's UI."
        ),
    )

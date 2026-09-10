"""Feishu/Lark native message_cot transport.

Only user-visible commentary and redacted tool summaries enter this channel. Provider
reasoning fields are intentionally outside this module's API.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
import threading
import time
from typing import Any, Awaitable, Callable
import uuid
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

COT_TEXT_MAX = 1200
COT_TOOL_OUTPUT_MAX = 1200
COT_FLUSH_INTERVAL_SECONDS = 0.6


def _tool_icon(name: str) -> str:
    lower = name.lower()
    for marker, icon in (
        (("search", "grep", "rg"), "search"),
        (("read",), "read"),
        (("write", "edit", "patch"), "write"),
        (("doc",), "doc"),
        (("calendar",), "calendar"),
        (("task",), "task"),
        (("command", "bash", "terminal"), "bash"),
    ):
        if any(part in lower for part in marker):
            return icon
    return "default"


def normalize_cot_mode(value: Any) -> str:
    if value is None or value is False:
        return "off"
    if value is True:
        return "brief"
    normalized = str(value).strip().lower()
    if normalized in {"off", "false", "no", "0"}:
        return "off"
    if normalized in {"detailed", "verbose"}:
        return "detailed"
    return (
        "brief"
        if normalized in {"brief", "simple", "on", "true", "yes", "1"}
        else "off"
    )


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _safe_text(value: Any, limit: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    from gateway.run import _redact_gateway_user_facing_secrets

    return _truncate(_redact_gateway_user_facing_secrets(text), limit)


class FeishuCOTError(RuntimeError):
    def __init__(self, code: Any = None):
        super().__init__("Feishu message_cot request failed")
        self.code = code


class FeishuCOTClient:
    def __init__(
        self,
        app_id: str,
        app_secret: str,
        domain: str,
        *,
        request_json: Callable[[str, str, dict | None], Awaitable[dict]] | None = None,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._base_url = (
            "https://open.larksuite.com"
            if domain == "lark"
            else "https://open.feishu.cn"
        )
        self._request_override = request_json
        self._token = ""
        self._token_expires_at = 0.0
        self._token_lock = asyncio.Lock()

    async def _http_json(
        self, method: str, path: str, body: dict | None, token: str = ""
    ) -> dict:
        def request() -> dict:
            headers = {"Content-Type": "application/json; charset=utf-8"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            data = (
                json.dumps(body, ensure_ascii=False).encode("utf-8")
                if body is not None
                else None
            )
            try:
                with urlopen(
                    Request(
                        self._base_url + path, data=data, headers=headers, method=method
                    ),
                    timeout=10,
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    status = response.status
            except HTTPError as exc:
                raise FeishuCOTError(exc.code) from exc
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise FeishuCOTError() from exc
            if not isinstance(payload, dict):
                raise FeishuCOTError(status)
            if status >= 400 or payload.get("code", 0) != 0:
                raise FeishuCOTError(payload.get("code", status))
            return payload

        return await asyncio.to_thread(request)

    async def _tenant_token(self) -> str:
        if self._token and time.monotonic() < self._token_expires_at:
            return self._token
        async with self._token_lock:
            if self._token and time.monotonic() < self._token_expires_at:
                return self._token
            data = await self._http_json(
                "POST",
                "/open-apis/auth/v3/tenant_access_token/internal",
                {"app_id": self._app_id, "app_secret": self._app_secret},
            )
            token = str(data.get("tenant_access_token") or "")
            if not token:
                raise FeishuCOTError(data.get("code"))
            expires_in = max(float(data.get("expire") or 7200), 120.0)
            self._token, self._token_expires_at = (
                token,
                time.monotonic() + expires_in - 60,
            )
            return token

    async def request_json(
        self, method: str, path: str, body: dict | None = None
    ) -> dict:
        if self._request_override is not None:
            return await self._request_override(method, path, body)
        return await self._http_json(method, path, body, await self._tenant_token())

    @staticmethod
    def _log_failure(stage: str, exc: Exception) -> None:
        code = getattr(exc, "code", None)
        logger.warning(
            "[Feishu COT] %s failed: type=%s code=%s",
            stage,
            type(exc).__name__,
            code if code is not None else "unknown",
        )

    async def start(
        self,
        chat_id: str,
        origin_message_id: str | None,
        mode: str,
        input_preview: str = "",
        *,
        flush_interval: float = COT_FLUSH_INTERVAL_SECONDS,
    ) -> "FeishuCOTRun | None":
        body = {"receive_id": chat_id}
        if origin_message_id:
            body["origin_message_id"] = origin_message_id
        try:
            data = await self.request_json(
                "POST",
                "/open-apis/im/v1/message_cot?receive_id_type=chat_id",
                body,
            )
            result = data.get("data") or data
            cot_id, message_id = (
                str(result.get("cot_id") or ""),
                str(result.get("message_id") or ""),
            )
            if not cot_id or not message_id:
                raise FeishuCOTError(data.get("code"))
            return FeishuCOTRun(
                self,
                cot_id,
                message_id,
                normalize_cot_mode(mode),
                flush_interval,
                chat_id=chat_id,
                input_preview=input_preview,
            )
        except Exception as exc:
            self._log_failure("create", exc)
            return None

    async def close(self) -> None:
        return None


class FeishuCOTRun:
    def __init__(
        self,
        client: FeishuCOTClient,
        cot_id: str,
        message_id: str,
        mode: str,
        flush_interval: float,
        *,
        chat_id: str,
        input_preview: str,
    ) -> None:
        self._client = client
        self.cot_id = cot_id
        self.message_id = message_id
        self.mode = mode
        self.run_id = uuid.uuid4().hex
        self.thread_id = chat_id
        self._flush_interval = flush_interval
        self._loop = asyncio.get_running_loop()
        self._events: list[dict] = []
        self._flush_task: asyncio.Task | None = None
        self._flush_lock = asyncio.Lock()
        self._finished = False
        self._timestamp_lock = threading.Lock()
        self._last_timestamp = 0
        self._active_step: str | None = f"step-start-{self.run_id}"
        self._active_step_name = "Agent 正在执行"
        self._tool_summaries: dict[str, str] = {}
        self._disabled = False
        self._append_event(
            "RUN_STARTED",
            {
                "threadId": self.thread_id,
                "runId": self.run_id,
                "input": {"query": _safe_text(input_preview, COT_TEXT_MAX)},
            },
        )
        self._append_event(
            "STEP_STARTED",
            {"stepId": self._active_step, "stepName": self._active_step_name},
        )

    def _timestamp(self) -> int:
        with self._timestamp_lock:
            now = int(time.time() * 1000)
            self._last_timestamp = max(now, self._last_timestamp + 1)
            return self._last_timestamp

    def _append_event(self, event_type: str, payload: dict) -> None:
        if self._finished or self._disabled:
            return
        self._events.append({
            "event_type": event_type,
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            "timestamp": self._timestamp(),
        })
        if self._flush_task is None:
            self._flush_task = self._loop.create_task(self._flush_later())

    def _emit(self, event_type: str, payload: dict) -> None:
        self._loop.call_soon_threadsafe(self._append_event, event_type, payload)

    async def _flush_later(self) -> None:
        try:
            await asyncio.sleep(self._flush_interval)
            self._flush_task = None
            await self.flush()
        except asyncio.CancelledError:
            pass

    async def flush(self) -> None:
        async with self._flush_lock:
            events, self._events = self._events, []
            if not events:
                return
            try:
                await self._client.request_json(
                    "PUT",
                    "/open-apis/im/v1/message_cot",
                    {
                        "cot_id": self.cot_id,
                        "message_id": self.message_id,
                        "events": events,
                    },
                )
            except Exception as exc:
                self._client._log_failure("update", exc)
                self._disabled = True
                self._events.clear()

    def step(self, iteration: int, tools: list | None = None) -> None:
        if self._active_step is not None:
            self._emit(
                "STEP_FINISHED",
                {"stepId": self._active_step, "stepName": self._active_step_name},
            )
        self._active_step = f"step-{self.run_id}-{iteration}"
        self._active_step_name = f"执行阶段 {iteration}"
        self._emit(
            "STEP_STARTED",
            {"stepId": self._active_step, "stepName": self._active_step_name},
        )

    def tool_started(self, call_id: str, tool_name: str, args: dict) -> None:
        from agent.display import build_tool_preview

        preview = _safe_text(
            build_tool_preview(tool_name, args or {}, max_len=120) or "", 120
        )
        call_key = str(call_id)
        title = preview or str(tool_name)
        self._tool_summaries[call_key] = title
        self._emit(
            "TOOL_CALL_START",
            {
                "toolCallId": call_key,
                "icon": _tool_icon(str(tool_name)),
                "title": title,
                "toolCallName": str(tool_name),
            },
        )
        if self.mode == "detailed":
            self._emit(
                "TOOL_CALL_ARGS",
                {
                    "toolCallId": call_key,
                    "delta": _safe_text(args or {}, COT_TEXT_MAX),
                },
            )
        self._emit("TOOL_CALL_END", {"toolCallId": call_key})

    def tool_completed(
        self, call_id: str, tool_name: str, args: dict, result: Any
    ) -> None:
        from agent.display import _detect_tool_failure

        result_text = (
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False, default=str)
        )
        is_error, _ = _detect_tool_failure(tool_name, result_text)
        call_key = str(call_id)
        output = (
            _safe_text(result, COT_TOOL_OUTPUT_MAX)
            if self.mode == "detailed"
            else self._tool_summaries.get(call_key, str(tool_name))
            + ("（失败）" if is_error else "（完成）")
        )
        self._tool_summaries.pop(call_key, None)
        self._emit(
            "TOOL_CALL_RESULT",
            {
                "messageId": f"tool-result-{call_key}",
                "toolCallId": call_key,
                "role": "tool",
                "content": output,
            },
        )

    def commentary(self, text: str) -> None:
        visible = _safe_text(text, COT_TEXT_MAX).strip()
        if not visible:
            return
        message_id = f"text-{self._timestamp()}"
        self._emit("TEXT_MESSAGE_START", {"messageId": message_id, "role": "assistant"})
        self._emit("TEXT_MESSAGE_CONTENT", {"messageId": message_id, "delta": visible})
        self._emit("TEXT_MESSAGE_END", {"messageId": message_id})

    async def finish(self, reason: str = "done") -> None:
        if self._finished:
            return
        await asyncio.sleep(
            0
        )  # drain call_soon_threadsafe events from the agent worker
        if self._active_step is not None:
            self._append_event(
                "STEP_FINISHED",
                {"stepId": self._active_step, "stepName": self._active_step_name},
            )
        api_reason = "done" if reason == "done" else "error"
        self._append_event(
            "RUN_FINISHED" if api_reason == "done" else "RUN_ERROR",
            (
                {"threadId": self.thread_id, "runId": self.run_id, "status": "done"}
                if api_reason == "done"
                else {"message": reason, "code": reason}
            ),
        )
        task = self._flush_task
        if task is not None:
            task.cancel()
            self._flush_task = None
            with suppress(asyncio.CancelledError):
                await task
        await self.flush()
        self._finished = True
        if self._disabled:
            return
        path = (
            f"/open-apis/im/v1/message_cot/complete/{quote(self.cot_id, safe='')}"
            f"?message_id={quote(self.message_id, safe='')}&reason={api_reason}"
        )
        try:
            await self._client.request_json("POST", path)
        except Exception as exc:
            self._client._log_failure("complete", exc)

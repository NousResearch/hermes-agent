"""
WebSocket Server adapter — Hermes Agent 作为 WS 服务端，前端客户端主动连接。

让 hermes-webui（或其他前端）通过 WebSocket 直连 Hermes Agent，获得完整的
工具集访问（terminal, execute_code, browser, file ops 等）。

架构：
  前端 (Vue SPA) → HTTP/SSE → webui 后端 (FastAPI)
                                      ↓ WS 客户端
  Hermes Agent (WS Server Adapter) ←←←←←←←← 监听 :8765/_ws

对比 api_server adapter：
  - WS server 使用 aiohttp WebSocket，而非 HTTP SSE
  - 持久的全双工连接（支持 async_delivery）
  - 精简的消息协议（JSON over WS），无 OpenAI 兼容开销
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

import aiohttp
from aiohttp import web

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    SessionSource,
)

logger = logging.getLogger(__name__)

AIOHTTP_AVAILABLE = True

_HEARTBEAT_INTERVAL = 30  # seconds


class WSServerAdapter(BasePlatformAdapter):
    """
    WebSocket 服务端适配器。

    监听 TCP 端口，接受 WS 客户端连接。每条连接认证后绑定一个 chat_id，
    所有消息/事件通过 WS 帧传输。
    """

    supports_code_blocks = True
    splits_long_messages = True
    supports_async_delivery = True
    MAX_MESSAGE_LENGTH = 32000

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("ws_server"))
        extra = config.extra or {}

        self._host: str = (
            extra.get("host")
            or os.environ.get("WS_SERVER_HOST", "127.0.0.1")
        )
        self._port: int = int(
            extra.get("port")
            or os.environ.get("WS_SERVER_PORT", "8765")
        )
        self._api_key: str = (
            extra.get("api_key")
            or os.environ.get("WS_SERVER_API_KEY", "")
        )

        # aiohttp 组件
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

        # chat_id → WebSocketResponse 映射
        self._clients: dict[str, web.WebSocketResponse] = {}
        self._ws_close_events: dict[str, asyncio.Event] = {}

        # per-chat 串行锁
        self._chat_locks: dict[str, asyncio.Lock] = {}
        _CHAT_LOCKS_MAX = 1000

        # 审批状态
        self._pending_approvals: dict[str, dict] = {}
        self._approval_counter = 0

        # 后台任务跟踪
        self._background_tasks: set[asyncio.Task] = set()
        self._heartbeat_task: Optional[asyncio.Task] = None

        # 是否正在运行
        self._running = False

    # ════════════════════════════════════════════════════════════════
    # 平台生命周期
    # ════════════════════════════════════════════════════════════════

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """启动 aiohttp WebSocket 服务端。"""
        if not self._api_key:
            logger.error(
                "[ws_server] Refusing to start: WS_SERVER_API_KEY is required "
                "(set in config.yaml platforms.ws_server.extra.api_key or "
                "WS_SERVER_API_KEY env var). This endpoint dispatches "
                "terminal-capable agent work — a guessable key is RCE."
            )
            return False

        self._app = web.Application()

        # 路由
        self._app.router.add_get("/_ws", self._handle_websocket)
        self._app.router.add_get("/health", self._handle_health)
        self._app["ws_server_adapter"] = self

        # 端口冲突检测
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                _s.settimeout(1)
                _s.connect(("127.0.0.1", self._port))
            logger.error(
                "[ws_server] Port %d already in use. Set a different port "
                "via WS_SERVER_PORT or config.yaml",
                self._port,
            )
            return False
        except (ConnectionRefusedError, OSError):
            pass  # port is free

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

        # 后台心跳
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._background_tasks.add(self._heartbeat_task)

        self._running = True
        self._mark_connected()
        logger.info(
            "[ws_server] Listening on ws://%s:%d/_ws",
            self._host, self._port,
        )
        return True

    async def disconnect(self) -> None:
        """停止 WS 服务端，断开所有客户端连接。"""
        self._running = False
        self._mark_disconnected()

        # 断开所有 WS 客户端
        for chat_id, ws in list(self._clients.items()):
            try:
                await ws.close()
            except Exception:
                pass
        self._clients.clear()

        # 停止后台任务
        for task in list(self._background_tasks):
            task.cancel()
        self._background_tasks.clear()

        # 停止 aiohttp 服务
        if self._site:
            try:
                await self._site.stop()
            except Exception:
                pass
            self._site = None
        if self._runner:
            try:
                await self._runner.cleanup()
            except Exception:
                pass
            self._runner = None
        self._app = None
        logger.info("[ws_server] Server stopped")

    # ════════════════════════════════════════════════════════════════
    # HTTP / WS 端点
    # ════════════════════════════════════════════════════════════════

    async def _handle_health(self, request: web.Request) -> web.Response:
        """GET /health — 健康检查。"""
        return web.json_response({
            "status": "ok",
            "platform": "ws_server",
            "connected_clients": len(self._clients),
        })

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket 连接主处理。

        认证流程：
          1. 客户端连接后第一条消息必须是 {"type": "auth", "api_key": "..."}
          2. 服务端验证 api_key，回复 {"type": "auth_ok", "chat_id": "..."}
          3. 后续消息按 type 路由
        """
        ws = web.WebSocketResponse(
            max_msg_size=1024 * 1024,  # 1MB
            heartbeat=_HEARTBEAT_INTERVAL,
        )
        await ws.prepare(request)

        # ── 等待认证 ──────────────────────────────────────────────
        try:
            auth_msg = await ws.receive_json()
        except Exception:
            await ws.close(code=4001, message=b"auth required")
            return ws

        if not isinstance(auth_msg, dict) or auth_msg.get("type") != "auth":
            await ws.send_json({"type": "error", "message": "first message must be auth"})
            await ws.close(code=4001, message=b"auth required")
            return ws

        if auth_msg.get("api_key") != self._api_key:
            await ws.send_json({"type": "error", "message": "invalid api_key"})
            await ws.close(code=4001, message=b"invalid api_key")
            return ws

        # 分配/使用客户端指定的 chat_id
        chat_id = str(auth_msg.get("chat_id", "")) or f"ws_{uuid.uuid4().hex[:12]}"

        # 如果已有相同 chat_id 的连接，关闭旧的
        old_ws = self._clients.get(chat_id)
        if old_ws and not old_ws.closed:
            try:
                await old_ws.close(code=4000, message=b"replaced by new connection")
            except Exception:
                pass

        self._clients[chat_id] = ws
        await ws.send_json({"type": "auth_ok", "chat_id": chat_id})
        logger.info("[ws_server] Client '%s' authenticated (total: %d)",
                     chat_id, len(self._clients))

        # ── 消息接收循环 ──────────────────────────────────────────
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    await self._dispatch_ws_message(data, chat_id)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.warning("[ws_server] WS error for '%s': %s",
                                   chat_id, ws.exception())
                    break
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("[ws_server] WS receive error for '%s'", chat_id)
        finally:
            self._clients.pop(chat_id, None)
            logger.info("[ws_server] Client '%s' disconnected (remaining: %d)",
                         chat_id, len(self._clients))

        return ws

    async def _dispatch_ws_message(self, data: dict, chat_id: str) -> None:
        """根据 type 分发 WS 消息。"""
        t = data.get("type", "")

        if t == "ping":
            await self._ws_send(chat_id, {"type": "pong"})
        elif t == "pong":
            pass
        elif t == "msg":
            # 动态注册 chat_id → WS 连接映射。
            # 客户端可能在 auth 后通过不同 chat_id 发消息，
            # 如果不重新注册，adapter.send(chat_id) 会找不到连接。
            msg_chat_id = data.get("chat_id", chat_id)
            ws = self._clients.get(chat_id)
            if ws and msg_chat_id != chat_id:
                self._clients[msg_chat_id] = ws
            await self._handle_inbound(data, msg_chat_id)
        elif t == "approve":
            await self._handle_approval(data)
        elif t == "stop":
            logger.info("[ws_server] Stop signal for '%s': %s",
                        chat_id, data.get("run_id", ""))
        else:
            logger.debug("[ws_server] Unknown message type '%s' from '%s'", t, chat_id)

    # ════════════════════════════════════════════════════════════════
    # 入站消息处理
    # ════════════════════════════════════════════════════════════════

    async def _handle_inbound(self, msg: dict, chat_id: str) -> None:
        """处理用户消息 → 构造 MessageEvent → agent pipeline。"""
        text = msg.get("text", "")
        if not text:
            return

        user_id = str(msg.get("user_id", chat_id))
        username = str(msg.get("username", user_id))

        source = self.build_source(
            chat_id=chat_id,
            chat_name=username,
            chat_type="dm",
            user_id=user_id,
            user_name=username,
        )

        event = MessageEvent(
            text=text,
            message_type=MessageType.COMMAND if text.startswith("/") else MessageType.TEXT,
            source=source,
            raw_message=msg,
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )

        await self._handle_message_with_guards(event, chat_id)

    async def _handle_message_with_guards(self, event: MessageEvent, chat_id: str) -> None:
        """per-chat 串行锁保证同一 chat 的消息串行处理。"""
        lock = self._chat_locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._chat_locks[chat_id] = lock

        async with lock:
            await self.handle_message(event)

    # ════════════════════════════════════════════════════════════════
    # 审批处理
    # ════════════════════════════════════════════════════════════════

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[dict] = None,
    ) -> SendResult:
        """发送审批请求到前端。"""
        self._approval_counter += 1
        approval_id = str(self._approval_counter)

        self._pending_approvals[approval_id] = {
            "session_key": session_key,
            "chat_id": chat_id,
        }

        ok = await self._ws_send(chat_id, {
            "type": "approval_card",
            "chat_id": chat_id,
            "approval_id": approval_id,
            "command": command,
            "reason": description,
        })

        return SendResult(success=ok, message_id=approval_id)

    async def _handle_approval(self, msg: dict) -> None:
        """处理前端返回的审批结果。"""
        approval_id = str(msg.get("approval_id", ""))
        choice = msg.get("choice", "deny")

        state = self._pending_approvals.pop(approval_id, None)
        if not state:
            logger.warning("[ws_server] Approval %s not found (expired?)", approval_id)
            return

        session_key = state["session_key"]

        logger.info(
            "[ws_server] Approval %s resolved: %s (session_key=%s)",
            approval_id, choice, session_key,
        )

        try:
            from tools.approval import resolve_gateway_approval
            resolve_gateway_approval(session_key, choice)
        except Exception as exc:
            logger.error("[ws_server] Resolve approval failed: %s", exc)

    # ════════════════════════════════════════════════════════════════
    # 出站消息（BasePlatformAdapter 接口）
    # ════════════════════════════════════════════════════════════════

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> SendResult:
        ok = await self._ws_send(chat_id, {
            "type": "send",
            "chat_id": chat_id,
            "content": content,
        })
        return SendResult(success=ok, message_id=str(uuid.uuid4()))

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        ok = await self._ws_send(chat_id, {
            "type": "edit",
            "chat_id": chat_id,
            "message_id": message_id,
            "content": content,
        })
        return SendResult(success=ok, message_id=message_id)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> SendResult:
        ok = await self._ws_send(chat_id, {
            "type": "send_file",
            "chat_id": chat_id,
            "file_path": image_url,
            "caption": caption or "",
        })
        return SendResult(success=ok, message_id=str(uuid.uuid4()))

    async def send_image_file(
        self,
        chat_id: str,
        file_path: str,
        caption: str = "",
        **kwargs,
    ) -> SendResult:
        return await self.send_image(chat_id, file_path, caption)

    send_document = send_image_file
    send_voice = send_image_file
    send_video = send_image_file

    async def send_typing(self, chat_id: str, metadata: Optional[dict] = None) -> None:
        await self._ws_send(chat_id, {
            "type": "typing",
            "chat_id": chat_id,
        })

    async def on_processing_start(self, event: MessageEvent) -> None:
        """处理开始 → 发送 typing 指示。"""
        chat_id = (
            getattr(event.source, "chat_id", "") or ""
            if event.source else ""
        )
        await self.send_typing(chat_id)

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome
    ) -> None:
        """处理完成 → 发送 done 事件，失败时额外发送错误提示。"""
        chat_id = (
            getattr(event.source, "chat_id", "") or ""
            if event.source else ""
        )
        if outcome is ProcessingOutcome.FAILURE:
            await self._ws_send(chat_id, {
                "type": "system",
                "chat_id": chat_id,
                "message": "❌ Processing failed, please retry",
            })
        await self._ws_send(chat_id, {
            "type": "done",
            "chat_id": chat_id,
        })

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"name": chat_id, "type": "dm"}

    # ════════════════════════════════════════════════════════════════
    # 心跳
    # ════════════════════════════════════════════════════════════════

    async def _heartbeat_loop(self) -> None:
        """定期向所有客户端发送 ping。"""
        while self._running:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            for chat_id in list(self._clients.keys()):
                await self._ws_send(chat_id, {"type": "ping"})

    # ════════════════════════════════════════════════════════════════
    # 辅助方法
    # ════════════════════════════════════════════════════════════════

    async def _ws_send(self, chat_id: str, data: dict) -> bool:
        """向指定 chat 发送 JSON 消息。"""
        ws = self._clients.get(chat_id)
        if not ws or ws.closed:
            return False
        try:
            await ws.send_json(data, dumps=lambda o: json.dumps(o, ensure_ascii=False))
            return True
        except Exception as exc:
            logger.debug("[ws_server] Send to '%s' failed: %s", chat_id, exc)
            return False

    def _get_cron_sender(self) -> Optional[Callable]:
        """返回 cron 投递函数。"""
        return None  # WS server 暂不支持 cron 投递


# ════════════════════════════════════════════════════════════════════
# 模块级辅助
# ════════════════════════════════════════════════════════════════════

import socket as _socket


def _check_requirements() -> bool:
    """检查 aiohttp 是否可用。"""
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        logger.error("[ws_server] aiohttp not installed")
        return False


def _build_adapter(config: PlatformConfig) -> WSServerAdapter:
    return WSServerAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by Hermes plugin system."""
    ctx.register_platform(
        name="ws_server",
        label="WebSocket Server",
        adapter_factory=_build_adapter,
        check_fn=_check_requirements,
        is_connected=lambda: _check_requirements(),
        validate_config=lambda _cfg: _check_requirements(),
        required_env=["WS_SERVER_API_KEY"],
        install_hint="aiohttp is bundled with Hermes Agent",
        emoji="🔌",
        max_message_length=32000,
        allow_update_command=True,
    )
"""OneBot 11 platform adapter for Hermes Agent.

Connects Hermes to QQ through the OneBot 11 protocol, compatible with
NapCat / Lagrange / LLOneBot / go-cqhttp.

Transports:
- reverse (default): the adapter hosts a WebSocket server; NapCat's
  "ws-reverse" client dials in. Inbound events and outbound actions
  share that single connection.
- forward: the adapter dials NapCat's "ws" server (ws://host:port).

Configuration (config.yaml):

    gateway:
      platforms:
        onebot:
          enabled: true
          extra:
            mode: reverse              # reverse | forward
            host: "127.0.0.1"          # reverse: listen address
            port: 8643                 # reverse: listen port
            url: "ws://127.0.0.1:3001" # forward: NapCat ws endpoint
            access_token: ""           # optional OneBot access token
            bot_qq: ""                 # optional; auto-learned from meta events
            require_mention: true      # group chats: only reply when @'d
            dm_policy: open            # open | allowlist | disabled
            allow_from: []             # user ids when dm_policy=allowlist
            group_policy: open         # open | allowlist | disabled
            group_allow_from: []       # group ids when group_policy=allowlist
            hot_reload: false          # dev only: reload onebot_utils/t2i_render on mtime change
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import io
import json
import logging
import mimetypes
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import aiohttp
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - gateway always ships aiohttp
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    _AUDIO_EXTS,
)
from gateway.session import SessionSource

# ---------------------------------------------------------------------------
# OneBot 11 constants
# ---------------------------------------------------------------------------

DEFAULT_PORT = 8643
ACTION_TIMEOUT = 30.0
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]
MAX_RECONNECT_ATTEMPTS = 100

# Image extensions for the standalone HTTP sender (base.py keeps its own
# _IMAGE_EXTS as a function-local set, so we declare the plugin copy).
_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"})
IMAGE_MAX_BYTES = 8 * 1024 * 1024  # skip absurdly large CQ image downloads
MEDIA_MAX_BYTES = 20 * 1024 * 1024  # voice/video/file base64 cap (NapCat upload)
MAX_MESSAGE_LENGTH = 4000  # QQ per-message cap (UTF-16-ish, keep safe)

PLUGIN_VERSION = "1.0.0"

# Max bytes for a downloaded voice clip (silk/amr from QQ).
AUDIO_MAX_BYTES = 15 * 1024 * 1024

# Content longer than this is rendered as a text image instead of being
# sent as text (0 / negative disables the image path).
DEFAULT_TEXT_IMAGE_THRESHOLD = 150

_UTILS_MTIME: float = 0.0
_utils_lock = threading.Lock()
_HOT_RELOAD: bool = False  # 由 adapter 初始化时从 extra.hot_reload 读取


def _set_hot_reload(enabled: bool) -> None:
    """开关 mtime 热加载（生产部署建议关闭；开发迭代样式时开启）。"""
    global _HOT_RELOAD
    _HOT_RELOAD = bool(enabled)


def _load_onebot_utils():
    """加载 onebot_utils 模块；extra.hot_reload=true 时检测 mtime 变化自动 reload。

    热加载语义：onebot_utils.py 每次修改后（保存即生效），下一次调用
    自动使用新逻辑，无需重启 gateway。覆盖 CQ 解析、Markdown 剥离、
    长消息分段、表情映射等纯规则。默认关闭——生产环境避免升级/部署
    期间半写入文件触发 reload。
    """
    global _UTILS_MTIME
    try:
        from . import onebot_utils as mod
    except ImportError:  # 插件以裸模块方式加载时
        import onebot_utils as mod

    if not _HOT_RELOAD:
        return mod

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onebot_utils.py")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0

    if mtime and mtime != _UTILS_MTIME:
        with _utils_lock:
            if mtime != _UTILS_MTIME:
                try:
                    importlib.reload(mod)
                    logger.info("onebot_utils.py 热加载生效 (mtime=%s)", mtime)
                except Exception:
                    logger.exception("onebot_utils.py 热加载失败，沿用旧模块")
                _UTILS_MTIME = mtime
    return mod


_T2I_MTIME: float = 0.0
_t2i_lock = threading.Lock()


def _load_t2i_render():
    """加载 t2i_render 模块；extra.hot_reload=true 时检测 mtime 变化自动 reload。

    热加载语义：t2i_render.py 每次修改后（保存即生效），下一次渲染
    自动使用新样式，无需重启 gateway。reload 后模块级缓存
    （_FONT_CACHE / _EMOJI_BITMAP_CACHE 等）随之重建，不会用旧字号。
    默认关闭——生产环境避免升级/部署期间半写入文件触发 reload。
    """
    global _T2I_MTIME
    try:
        from . import t2i_render as mod
    except ImportError:  # 插件以裸模块方式加载时
        import t2i_render as mod

    if not _HOT_RELOAD:
        return mod

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2i_render.py")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0

    if mtime and mtime != _T2I_MTIME:
        with _t2i_lock:
            if mtime != _T2I_MTIME:
                try:
                    importlib.reload(mod)
                    logger.info("t2i_render.py 热加载生效 (mtime=%s)", mtime)
                except Exception:
                    logger.exception("t2i_render.py 热加载失败，沿用旧模块")
                _T2I_MTIME = mtime
    return mod


def render_text_image(text: str, title: Optional[str] = None) -> bytes:
    """Render *text* as a styled Markdown card image (AstrBot-style renderer).

    See t2i_render.py for the element-based Markdown renderer (bold/italic/
    headers/quotes/lists/code/table support) with glyph-level font fallback.
    ``title`` (e.g. "To 昵称") is drawn as a top bar on the card.

    热加载：每次调用检测 t2i_render.py 的 mtime，文件变化自动 reload，
    改样式无需重启 gateway。
    """
    mod = _load_t2i_render()
    return mod.render_text_image(text, title)


def _build_chat_id(message_type: str, id_: Any) -> str:
    """Canonical chat_id used by the session store and outbound sends."""
    return _load_onebot_utils()._build_chat_id(message_type, id_)


def _split_chat_id(chat_id: str) -> Tuple[str, str]:
    """Return (kind, target) — kind is 'private' or 'group'."""
    return _load_onebot_utils()._split_chat_id(chat_id)


class OneBotAdapter(BasePlatformAdapter):
    """QQ via OneBot 11 (NapCat etc.)."""

    # Per-message cap (QQ ~4000 chars). The gateway reads this class
    # attribute to split long responses into multiple messages.
    MAX_MESSAGE_LENGTH: int = MAX_MESSAGE_LENGTH

    @property
    def enforces_own_access_policy(self) -> bool:
        """本 adapter 在入站自行执行访问策略（dm/group allowlist + 角色分级）。

        gateway authz 在 effective policy 为 allowlist 时信任 adapter 的
        名单决策（按群号放行群成员、私聊仅管理员），不再用 env 白名单
        二次拦截放行的群消息。
        """
        return True

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("onebot"))
        extra = config.extra or {}
        # 开发迭代开关：hot_reload=true 时 onebot_utils/t2i_render 修改即生效
        _set_hot_reload(
            str(extra.get("hot_reload", False)).strip().lower() in {"true", "1", "yes"}
        )
        self._mode = str(extra.get("mode", "reverse")).strip().lower() or "reverse"
        self._host = str(extra.get("host", "127.0.0.1"))
        try:
            self._port = int(extra.get("port", DEFAULT_PORT))
        except (TypeError, ValueError):
            self._port = DEFAULT_PORT
        self._url = str(extra.get("url", "ws://127.0.0.1:3001"))
        self._access_token = str(extra.get("access_token", "") or "").strip()
        self._bot_qq = str(extra.get("bot_qq", "") or "").strip()
        _default_split = _load_onebot_utils().DEFAULT_SPLIT_LENGTH
        try:
            self._split_length = int(extra.get("split_length", _default_split))
        except (TypeError, ValueError):
            self._split_length = _default_split
        if self._split_length <= 0:
            self._split_length = _default_split
        try:
            self._text_image_threshold = int(
                extra.get("text_image_threshold", DEFAULT_TEXT_IMAGE_THRESHOLD)
            )
        except (TypeError, ValueError):
            self._text_image_threshold = DEFAULT_TEXT_IMAGE_THRESHOLD
        try:
            self._image_max_size = int(extra.get("image_max_size", 1536))
        except (TypeError, ValueError):
            self._image_max_size = 1536
        self._require_mention = bool(extra.get("require_mention", True))
        try:
            self._max_inbound_file_bytes = int(extra.get("max_inbound_file_bytes", MEDIA_MAX_BYTES))
        except (TypeError, ValueError):
            self._max_inbound_file_bytes = MEDIA_MAX_BYTES
        # 单条 interim 超时自动撤回（#2 回移，dsh 语义）：final 结算前每条
        # interim 独立计时，到时未结算就单独撤回；0 = 关闭（只靠 final 结算）
        try:
            self._interim_recall_seconds = float(extra.get("interim_recall_seconds", 90))
        except (TypeError, ValueError):
            self._interim_recall_seconds = 90.0
        self._dm_policy = str(extra.get("dm_policy", "open")).strip().lower()
        self._group_policy = str(extra.get("group_policy", "open")).strip().lower()
        self._allow_from = {str(v) for v in (extra.get("allow_from") or [])}
        self._group_allow_from = {str(v) for v in (extra.get("group_allow_from") or [])}

        # 权限分级：管理员集合（extra.admin_users 显式 > 回退 ONEBOT_ALLOWED_USERS）
        self._admin_users = {str(v) for v in (extra.get("admin_users") or [])}
        if not self._admin_users:
            self._admin_users = {
                u.strip()
                for u in os.environ.get("ONEBOT_ALLOWED_USERS", "").split(",")
                if u.strip()
            }

        # Runtime state
        self._ws: Optional[Any] = None  # live OneBot connection (read/write)
        self._self_id: Optional[str] = None  # bot's own QQ, learned from events
        self._member_chats: set = set()  # 普通用户受限会话（出站敏感审计用）
        self._nicknames: Dict[str, str] = {}  # chat_id -> last known user nickname
        self._load_nicknames()
        self._pending_actions: Dict[str, asyncio.Future] = {}
        self._runner: Optional[Any] = None  # reverse-mode web runner
        self._site: Optional[Any] = None
        self._forward_session: Optional[Any] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._stopping = False
        self._last_event_ts = 0.0
        # 一次回复周期内的中间消息缓冲: chat_id -> [(message_id, text), ...]
        # 收到最终回复（t2i 图片等）时合并为一条 QQ 转发并撤回原消息。
        self._loop_buffer: Dict[str, List[Tuple[str, str]]] = {}
        self._loop_buffer_ts: Dict[str, float] = {}
        # 合并转发已完成、待撤回的原消息 id（撤回在最终内容发送后执行）
        self._pending_recalls: Dict[str, List[str]] = {}
        # #6 回移：/mode per-chat 出站模式覆盖（False=instant 逐条即时）
        self._chat_interim_overrides: Dict[str, bool] = {}
        # #6 回移：/ocr 用的最近入站图片路径（per chat）
        self._last_image_path: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not AIOHTTP_AVAILABLE:
            logger.error("[onebot] aiohttp unavailable — cannot start adapter")
            return False
        if not self._acquire_platform_lock("onebot", self._mode, "OneBot mode"):
            return False
        self._stopping = False
        # #7 回移：t2i 墨水自检（防空豆腐卡；字体缺失仅告警不阻塞）
        try:
            if not is_reconnect:
                _ink = _load_t2i_render().ink_check()
                if not _ink.get("ok"):
                    logger.warning(
                        "[onebot] t2i ink check FAILED: loaded=%s cjk=%s — "
                        "long replies will render tofu/fall back to text",
                        _ink.get("loaded"), _ink.get("cjk"),
                    )
        except Exception as e:
            logger.debug("[onebot] t2i ink check skipped: %s", e)
        try:
            if self._mode == "forward":
                await self._connect_forward_once()
            else:
                await self._start_reverse_server()
            self._mark_connected()
            return True
        except Exception as e:
            logger.error("[onebot] connect failed: %s", e)
            self._mark_disconnected()
            return False

    async def disconnect(self) -> None:
        self._stopping = True
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        if self._forward_session is not None:
            try:
                await self._forward_session.close()
            except Exception:
                pass
            self._forward_session = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                pass
            self._runner = None
            self._site = None
        for fut in self._pending_actions.values():
            if not fut.done():
                fut.set_exception(ConnectionError("OneBot adapter disconnected"))
        self._pending_actions.clear()
        self._mark_disconnected()

    # -- reverse mode --------------------------------------------------

    async def _start_reverse_server(self) -> None:
        app = web.Application()
        app.router.add_get("/ws", self._handle_reverse_ws)
        app.router.add_get("/onebot", self._handle_reverse_ws)
        app.router.add_get("/", self._handle_reverse_ws)
        app.router.add_get("/api/group_history", self._handle_group_history)
        app.router.add_get("/api/napcat", self._handle_napcat_api)
        app.router.add_post("/api/send_media", self._handle_send_media)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        logger.info(
            "[onebot] reverse WS server listening on ws://%s:%s/ws (NapCat ws-reverse → this URL)",
            self._host, self._port,
        )

    async def _handle_group_history(self, request: web.Request) -> web.Response:
        """Local helper endpoint: pull group message history via NapCat API.

        GET /api/group_history?group_id=123456789&count=20[&message_seq=N]
        Reuses the reverse-WS echo mechanism, so it works without an HTTP API
        on the NapCat side. No auth (loopback/LAN only) — same posture as /ws.
        """
        try:
            group_id = int(request.query.get("group_id", "0") or "0")
            count = int(request.query.get("count", "20") or "20")
            seq_raw = request.query.get("message_seq")
            if group_id <= 0:
                return web.json_response({"status": "error", "error": "group_id required"}, status=400)
            params = {"group_id": group_id, "count": max(1, min(count, 50))}
            if seq_raw:
                params["message_seq"] = int(seq_raw)
            data = await self._call_action("get_group_msg_history", params, timeout=15.0)
            return web.json_response({"status": "ok", "data": data})
        except Exception as exc:
            return web.json_response({"status": "error", "error": str(exc)}, status=500)

    async def _handle_napcat_api(self, request: web.Request) -> web.Response:
        """Whitelisted NapCat action proxy for the qq_napcat_api tool.

        GET /api/napcat?action=<action>&params=<urlencoded-json>
        Same loopback/LAN posture as /api/group_history (no auth).
        """
        action = request.query.get("action", "")
        from plugins.platforms.onebot.tools import NAPCAT_API_WHITELIST

        if action not in NAPCAT_API_WHITELIST:
            return web.json_response(
                {"status": "error", "error": f"action {action!r} not whitelisted"},
                status=403,
            )
        try:
            params = json.loads(request.query.get("params") or "{}")
        except json.JSONDecodeError:
            params = {}
        try:
            data = await self._call_action(action, params, timeout=30.0)
        except Exception as exc:
            return web.json_response({"status": "error", "error": str(exc)}, status=500)
        return web.json_response({"status": "ok", "data": data})

    async def _handle_send_media(self, request: web.Request) -> web.Response:
        """Media-forwarding endpoint for the qq_send_* tools.

        POST /api/send_media  json: {chat_id, kind, ...}
          - kind=image:   {sources: [path|url], caption?}
          - kind=voice:   {path}
          - kind=video:   {path}
          - kind=file:    {path, file_name?}
          - kind=forward: {nodes: [{name, content}]}
        """
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"status": "error", "error": "bad json"}, status=400)
        chat_id = str(payload.get("chat_id") or "")
        kind = str(payload.get("kind") or "")
        if not chat_id or kind not in ("image", "voice", "video", "file", "forward"):
            return web.json_response(
                {"status": "error", "error": "chat_id and kind (image/voice/video/file/forward) required"},
                status=400,
            )
        try:
            message_id = None
            if kind == "image":
                sources = [s for s in (payload.get("sources") or []) if isinstance(s, str) and s]
                if not sources:
                    return web.json_response({"status": "error", "error": "sources empty"}, status=400)
                images = [
                    (s if s.startswith(("http://", "https://")) else f"file://{s}", "")
                    for s in sources
                ]
                await self.send_multiple_images(chat_id=chat_id, images=images)
            elif kind == "voice":
                path = str(payload.get("path") or "")
                if not path:
                    return web.json_response({"status": "error", "error": "path required"}, status=400)
                res = await self.send_voice(chat_id, path)
                message_id = getattr(res, "message_id", None)
                if not getattr(res, "success", False):
                    return web.json_response({"status": "error", "error": res.error or "send failed"})
            elif kind == "video":
                path = str(payload.get("path") or "")
                if not path:
                    return web.json_response({"status": "error", "error": "path required"}, status=400)
                res = await self.send_video(chat_id, path)
                message_id = getattr(res, "message_id", None)
                if not getattr(res, "success", False):
                    return web.json_response({"status": "error", "error": res.error or "send failed"})
            elif kind == "file":
                path = str(payload.get("path") or "")
                if not path:
                    return web.json_response({"status": "error", "error": "path required"}, status=400)
                res = await self.send_document(
                    chat_id, path, file_name=str(payload.get("file_name") or "") or None
                )
                message_id = getattr(res, "message_id", None)
                if not getattr(res, "success", False):
                    return web.json_response({"status": "error", "error": res.error or "send failed"})
            else:  # forward
                nodes = payload.get("nodes") or []
                if not isinstance(nodes, list) or not nodes:
                    return web.json_response({"status": "error", "error": "nodes empty"}, status=400)
                kind_cid, target = _split_chat_id(chat_id)
                array_nodes = [
                    {
                        "uin": self._self_id or "0",
                        "name": str(n.get("name") or "Hermes")[:20],
                        "content": [{"type": "text", "data": {"text": str(n.get("content") or "")[:500]}}],
                    }
                    for n in nodes
                    if isinstance(n, dict)
                ]
                if not array_nodes:
                    return web.json_response({"status": "error", "error": "nodes empty"}, status=400)
                if kind_cid == "group":
                    await self._call_action(
                        "send_forward_msg", {"group_id": int(target), "messages": array_nodes}, timeout=30.0
                    )
                else:
                    await self._call_action(
                        "send_private_forward_msg",
                        {"user_id": int(target), "messages": array_nodes},
                        timeout=30.0,
                    )
            return web.json_response({"status": "ok", "message_id": message_id})
        except Exception as exc:
            logger.warning("[onebot] /api/send_media failed: %s", exc)
            return web.json_response({"status": "error", "error": str(exc)}, status=500)

    async def _handle_reverse_ws(self, request: web.Request) -> web.WebSocketResponse:
        # Optional auth: NapCat sends `Authorization: Bearer <token>` when an
        # access token is configured on its side.
        if self._access_token:
            auth = request.headers.get("Authorization", "")
            expected = f"Bearer {self._access_token}"
            if auth != expected:
                logger.warning("[onebot] reverse WS auth rejected")
                return web.Response(status=401, text="unauthorized")
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        self._ws = ws
        logger.info("[onebot] NapCat connected via reverse WS")
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    self._handle_frame(data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.warning("[onebot] reverse WS error: %s", ws.exception())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("[onebot] reverse WS loop ended: %s", e)
        finally:
            if self._ws is ws:
                self._ws = None
            logger.info("[onebot] NapCat disconnected (reverse WS)")
        return ws

    # -- forward mode ---------------------------------------------------

    async def _connect_forward_once(self) -> None:
        headers = {"Authorization": f"Bearer {self._access_token}"} if self._access_token else {}
        session = aiohttp.ClientSession()
        try:
            ws = await session.ws_connect(
                self._url, headers=headers, heartbeat=30,
                timeout=aiohttp.ClientWSTimeout(ws_close=10.0),
            )
        except Exception:
            await session.close()
            raise
        self._forward_session = session
        self._ws = ws
        logger.info("[onebot] forward WS connected to %s", self._url)
        self._reader_task = asyncio.create_task(self._forward_read_loop(ws))

    async def _forward_read_loop(self, ws) -> None:
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    self._handle_frame(data)
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("[onebot] forward WS read error: %s", e)
        finally:
            if self._ws is ws:
                self._ws = None
            # 断线快速失败：挂起的 action future 立即报错，避免调用方
            # 干等 wait_for 超时（10-30s），并防止 future 泄漏
            if self._pending_actions:
                for fut in self._pending_actions.values():
                    if not fut.done():
                        fut.set_exception(
                            ConnectionError("OneBot WebSocket closed while awaiting action")
                        )
                self._pending_actions.clear()
            # 关闭本连接持有的 session，防止 aiohttp 连接泄漏
            sess = self._forward_session
            if sess is not None:
                self._forward_session = None
                try:
                    await sess.close()
                except Exception:
                    pass
            logger.info("[onebot] forward WS closed")
            if not self._stopping:
                if self._reconnect_task is None or self._reconnect_task.done():
                    self._reconnect_task = asyncio.create_task(self._forward_reconnect())

    async def _forward_reconnect(self) -> None:
        for delay in RECONNECT_BACKOFF * (MAX_RECONNECT_ATTEMPTS // len(RECONNECT_BACKOFF) + 1):
            if self._stopping:
                return
            await asyncio.sleep(delay)
            if self._stopping:
                return
            try:
                await self._connect_forward_once()
                logger.info("[onebot] forward WS reconnected")
                return
            except Exception as e:
                logger.warning("[onebot] forward WS reconnect failed: %s", e)

    # ------------------------------------------------------------------
    # Frame handling (shared by both transports)
    # ------------------------------------------------------------------

    def _handle_frame(self, data: dict) -> None:
        """Route an inbound OneBot frame: action response or event."""
        echo = data.get("echo")
        if echo is not None:
            fut = self._pending_actions.get(str(echo))
            if fut is not None and not fut.done():
                fut.set_result(data)
            return
        post_type = data.get("post_type")
        if post_type == "meta_event":
            self._learn_self_id(data.get("self_id"))
            return
        if post_type == "message":
            asyncio.create_task(self._process_message(data))
        # notice / request events are intentionally ignored for now.

    def _learn_self_id(self, self_id) -> None:
        if self_id is None:
            return
        sid = str(self_id)
        if self._self_id is None or self._self_id == sid:
            self._self_id = sid

    # -- 昵称持久化（t2i 顶栏；重启后 cron 推送/会话恢复也能画出顶栏） --

    def _nicknames_file(self) -> str:
        # Runtime state lives under HERMES_HOME, not the plugin package dir
        # (read-only pip installs / profile isolation).
        from hermes_constants import get_hermes_home

        return str(get_hermes_home() / "onebot_nicknames.json")

    def _load_nicknames(self) -> None:
        try:
            with open(self._nicknames_file(), "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._nicknames = {str(k): str(v) for k, v in data.items()}
        except Exception:
            self._nicknames = {}

    def _persist_nicknames(self) -> None:
        try:
            path = self._nicknames_file()
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._nicknames, f, ensure_ascii=False, indent=1)
            os.replace(tmp, path)
        except Exception as e:
            logger.debug("[onebot] persist nicknames failed: %s", e)

    # ------------------------------------------------------------------
    # Inbound messages
    # ------------------------------------------------------------------

    async def _process_message(self, data: dict) -> None:
        try:
            message_type = data.get("message_type", "")
            user_id = str(data.get("user_id", "") or "")
            self._learn_self_id(data.get("self_id"))
            raw = data.get("raw_message", "") or ""
            message = data.get("message")
            sender = data.get("sender") or {}
            nickname = sender.get("card") or sender.get("nickname") or ""

            if message_type == "private":
                if not self._dm_allowed(user_id):
                    return
                chat_id = _build_chat_id("private", user_id)
                chat_type = "dm"
            elif message_type == "group":
                group_id = str(data.get("group_id", "") or "")
                if not self._group_allowed(group_id):
                    return
                if self._require_mention and not self._is_mentioned(raw, message):
                    return
                chat_id = _build_chat_id("group", group_id)
                chat_type = "group"
            else:
                return

            # ── 权限分级（2026-08-13）────────────────────────────────
            # admin（extra.admin_users / ONEBOT_ALLOWED_USERS）：全权限
            # member（群内其他成员）：受限——注入标记 + 禁斜杠命令；私聊直接拒
            role = _load_onebot_utils().classify_user_role(user_id, self._admin_users)
            if chat_type == "dm" and role != "admin":
                logger.info("[onebot] dm from non-admin user %s rejected", user_id)
                return

            # 记录最近发言者昵称（文字图顶栏 "To XXX" 用）
            if nickname:
                self._nicknames[chat_id] = nickname
                self._persist_nicknames()

            # 新一轮用户消息: 清理上一轮残留的 loop 缓冲（防止跨轮合并）
            self._loop_buffer.pop(chat_id, None)
            self._loop_buffer_ts.pop(chat_id, None)
            self._pending_recalls.pop(chat_id, None)

            message = data.get("message")
            reply_id: Optional[str] = None
            if isinstance(message, list) and message:
                text, media_urls, media_types, reply_id = await self._parse_message_array(message)
            else:
                text, media_urls, media_types = await self._parse_content(raw)
                # CQ 字符串路径: 提取 [CQ:reply,id=xxx]
                rm = _load_onebot_utils()._CQ_REPLY_RE.search(raw)
                if rm:
                    reply_id = rm.group(1)

            # 普通用户斜杠命令拦截（/new /model /help /reset 等全禁）
            # 复用 get_command 同款规则：首词 /xxx 且命令名不含 /（排除路径误判）
            # @提及会拼在文本前（如 "@123456789/help"），先剥离开头 at 再判
            if role == "member" and text:
                _probe = re.sub(r"^@\d+\s*", "", text.lstrip())
                if _probe.startswith("/"):
                    cmd = _probe.split(maxsplit=1)[0][1:].lower()
                    if cmd and "/" not in cmd:
                        logger.info(
                            "[onebot] restricted user %s slash-command blocked: /%s",
                            user_id, cmd,
                        )
                        return  # 事件不构造，基类/run.py 无从分发

            # 普通用户会话记录到出站敏感审计集合
            if role == "member":
                self._member_chats.add(chat_id)

            # #6 回移：admin 本地命令（/ocr /mode /id /ver）直接处理回复，
            # 不构造事件给 agent；其余斜杠交网关原生分发
            if role == "admin" and text:
                _probe = re.sub(r"^@\d+\s*", "", text.lstrip())
                if _probe.startswith("/"):
                    cmd = _probe.split(maxsplit=1)[0][1:].lower()
                    if cmd and "/" not in cmd and cmd in ("ocr", "mode", "id", "ver"):
                        try:
                            reply = await self._handle_local_command(
                                chat_id, chat_type, user_id, _probe
                            )
                        except Exception as e:
                            logger.warning("[onebot] local command failed: %s", e)
                            reply = f"❌ 命令执行失败：{e}"
                        if reply:
                            await self.send(chat_id, reply)
                            return

            # 群聊普通用户：注入受限标记（agent 侧软限制依据）
            if role == "member" and chat_type == "group" and text:
                text = f"[受限用户:仅问答]\n{text}"

            # 用户引用了一条消息时, 从原消息取图片和文本（引用就是给 agent 看的）
            if reply_id:
                try:
                    orig = await self._call_action(
                        "get_msg", {"message_id": int(reply_id)}, timeout=10.0
                    )
                    orig_msg = orig.get("message")
                    if isinstance(orig_msg, list):
                        om_text, om_urls, om_types, _ = await self._parse_message_array(orig_msg)
                    elif orig_msg:
                        om_text, om_urls, om_types = await self._parse_content(str(orig_msg))
                    else:
                        om_text, om_urls, om_types = "", [], []
                    if om_text:
                        text = f"[引用]{om_text}\n{text}".strip()
                    if om_urls:
                        media_urls.extend(om_urls)
                        media_types.extend(om_types)
                except Exception as e:
                    logger.info("[onebot] get_msg failed for reply id=%s: %s", reply_id, e)

            has_voice = any(t.startswith("audio/") for t in media_types)
            # #6 回移：记录最近入站图片路径（/ocr 用，per chat）
            for i, mt in enumerate(media_types):
                if mt == "image" and i < len(media_urls) and media_urls[i]:
                    self._last_image_path[chat_id] = media_urls[i]
            if not text and not media_urls:
                return
            event = MessageEvent(
                text=text,
                message_type=(
                    MessageType.VOICE
                    if has_voice and not text
                    else MessageType.PHOTO
                    if media_urls
                    else MessageType.TEXT
                ),
                user_id=user_id,
                user_name=nickname or user_id,
                source=SessionSource(
                    platform=Platform("onebot"),
                    chat_id=chat_id,
                    chat_type=chat_type,
                    user_id=user_id,
                    user_name=nickname or user_id,
                ),
                raw_message=data,
                message_id=str(data.get("message_id") or ""),
                media_urls=media_urls,
                media_types=media_types,
            )
            await self.handle_message(event)
            # 顺带清理过期临时媒体文件（防 /tmp/hermes_onebot 无限堆积）
            self._cleanup_tmp_files()
        except Exception as e:
            logger.error("[onebot] failed processing message: %s", e, exc_info=True)

    def _dm_allowed(self, user_id: str) -> bool:
        policy = self._dm_policy
        if policy == "disabled":
            return False
        if policy == "allowlist":
            return user_id in self._allow_from
        # open：默认仅管理员可用私聊（与文档一致）。设置
        # ONEBOT_ALLOW_ALL_USERS=true / GATEWAY_ALLOW_ALL_USERS=true 时
        # 才是真正开放，否则该 opt-in 在 adapter 入站就被拦掉，
        # gateway 端的 allow-all 判定永远不会生效。
        if self._allow_all_users():
            return True
        return user_id in self._admin_users

    def _allow_all_users(self) -> bool:
        """allow-all opt-in（与 gateway authz 同语义：true/1/yes）"""
        for var in ("ONEBOT_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS"):
            if os.getenv(var, "").strip().lower() in {"true", "1", "yes"}:
                return True
        return False

    def _group_allowed(self, group_id: str) -> bool:
        policy = self._group_policy
        if policy == "disabled":
            return False
        if policy == "allowlist":
            return group_id in self._group_allow_from
        return True

    def _is_mentioned(self, raw: str, message: Optional[list] = None) -> bool:
        """True when the bot was @'d or the message replies to something.

        Prefer the structured message array (OneBot 11 default); fall back
        to CQ string parsing for text-format clients.
        With an unknown bot id and no configured bot_qq we fail closed in
        group chats (no accidental reply to every message).
        """
        self_id = self._self_id or self._bot_qq or ""
        if message is not None and isinstance(message, list):
            for seg in message:
                if not isinstance(seg, dict):
                    continue
                if seg.get("type") == "at" and str(
                    (seg.get("data") or {}).get("qq", "")
                ) == self_id:
                    return True
                if seg.get("type") == "reply":
                    # Replying to a message is an explicit nudge — treat as a call.
                    return True
            return False
        if self_id and f"[CQ:at,qq={self_id}]" in raw:
            return True
        if "[CQ:reply" in raw:
            # Replying to a message is an explicit nudge — treat as a call.
            return True
        return False

    async def _handle_local_command(self, chat_id: str, chat_type: str, user_id: str, text: str) -> Optional[str]:
        """#6 回移：admin 本地斜杠命令（/ocr /mode /id /ver）。

        返回要发送的回复文本；不认识的命令返回 None（交给网关斜杠分发）。
        """
        m = re.match(r"/(\S+)(?:\s+(.*))?$", text.strip())
        if not m:
            return None
        cmd = m.group(1).lower()
        arg = (m.group(2) or "").strip()
        if cmd == "id":
            return f"chat_id: {chat_id}\nuser_id: {user_id}"
        if cmd == "ver":
            return f"onebot-plugin v{PLUGIN_VERSION}"
        if cmd == "mode":
            if not arg:
                cur = self._chat_interim_overrides.get(chat_id)
                label = "interim（合并卡片）" if cur else "instant（逐条即时）"
                state = "（/mode 覆盖）" if cur is not None else "（默认 interim）"
                return f"当前出站模式：{label}{state}\n用法：/mode interim|instant"
            if arg in ("interim", "on", "merge"):
                self._chat_interim_overrides[chat_id] = True
                return "✅ 已切换为 interim（合并卡片）模式。下一条回复生效。"
            if arg in ("instant", "off", "direct"):
                self._chat_interim_overrides[chat_id] = False
                return "✅ 已切换为 instant（逐条即时）模式。下一条回复生效。"
            return "用法：/mode interim|instant"
        if cmd == "ocr":
            path = self._last_image_path.get(chat_id, "")
            if not path:
                return "请先在对话里发一张图片，再 /ocr。"
            try:
                b64 = await self._file_to_base64(path)
                if not b64:
                    return "❌ 读取图片失败（文件缺失或超限）。"
                data = await self._call_action(
                    "ocr_image", {"image": f"base64://{b64}"}, timeout=30.0
                )
                texts = [
                    t.get("text", "")
                    for t in (data.get("texts") or [])
                    if t.get("text")
                ]
                lines = "\n".join(texts).strip()
                if not lines:
                    return "OCR 未识别到文本。"
                return "OCR 结果：\n" + lines[:1500]
            except Exception as e:
                return f"❌ OCR 失败：{e}"
        return None

    async def _parse_content(self, raw: str) -> Tuple[str, List[str], List[str]]:
        """Convert a raw CQ-encoded message to (text, media_paths, media_types).

        Images with a downloadable url are fetched into a temp dir so the
        vision tool can read them; voice clips are downloaded and converted
        to 16 kHz mono WAV so the gateway's STT pipeline can transcribe
        them. Replies/at are normalized to plain text.
        """
        image_urls = _load_onebot_utils()._CQ_IMAGE_RE.findall(raw)
        media_urls: List[str] = []
        media_types: List[str] = []
        # 完整解析每个 CQ:image 的 url/file 属性（url 可能为空，需 get_image 换取）
        for m in _load_onebot_utils()._CQ_IMAGE_ALL_RE.finditer(raw):
            attrs = {}
            for kv in m.group(1).split(","):
                if "=" in kv:
                    k, _, v = kv.partition("=")
                    attrs[k.strip()] = _load_onebot_utils()._cq_unescape(v)
            logger.info("[onebot] CQ:image attrs: %s", attrs)
            try:
                path = await self._resolve_image(attrs.get("url", ""), attrs.get("file", ""))
                if path:
                    media_urls.append(path)
                    media_types.append("image")
            except Exception as e:
                logger.debug("[onebot] image download failed: %s", e)
        # 语音：优先 url 直下；NapCat 私聊语音常无 url（只有 file hash + 容器
        # 内 path），用 get_record API 以 file hash 换取 base64（2026-08-14）
        for m in _load_onebot_utils()._CQ_RECORD_ALL_RE.finditer(raw):
            attrs = {}
            for kv in m.group(1).split(","):
                if "=" in kv:
                    k, _, v = kv.partition("=")
                    attrs[k.strip()] = _load_onebot_utils()._cq_unescape(v)
            url = attrs.get("url", "")
            if not url and attrs.get("file"):
                try:
                    data = await self._call_action(
                        "get_record",
                        {"file": attrs["file"], "out_format": "wav"},
                        timeout=15.0,
                    )
                    url = data.get("url") or data.get("file") or ""
                except Exception as e:
                    logger.debug("[onebot] get_record failed for %s: %s", attrs.get("file"), e)
            if not url:
                continue
            try:
                path = await self._download_audio(url)
                if path:
                    media_urls.append(path)
                    media_types.append("audio/wav")
            except Exception as e:
                logger.debug("[onebot] voice download failed: %s", e)

        u = _load_onebot_utils()
        text = u._CQ_IMAGE_RE.sub(lambda m: "[图片]", raw)
        text = u._CQ_IMAGE_NOURL_RE.sub("[图片]", text)
        text = u._CQ_RECORD_RE.sub(lambda m: "[语音]", text)
        text = u._CQ_RECORD_NOURL_RE.sub("[语音]", text)
        text = u._CQ_AT_RE.sub(
            lambda m: "@" + ("全体成员" if m.group(1) == "all" else m.group(1)), text
        )
        text = u._CQ_REPLY_RE.sub(lambda m: "", text)
        text = u._CQ_FACE_RE.sub(lambda m: u._FACE_EMOJI.get(m.group(1), "[表情]"), text)
        # 文件：优先下载（CDN 直链 / get_file base64-url 双通道），成功注入
        # [文件:本地路径]，失败留待 _CQ_FILE_RE.sub 显示 [文件:名]
        for m in u._CQ_FILE_RE.finditer(raw):
            attrs = {}
            for kv in m.group(1).split(","):
                if "=" in kv:
                    k, _, v = kv.partition("=")
                    attrs[k.strip()] = u._cq_unescape(v.strip())
            try:
                fpath = await self._resolve_file(attrs)
            except Exception as e:
                logger.debug("[onebot] CQ file resolve failed: %s", e)
                fpath = None
            if fpath:
                text = text.replace(m.group(0), f"[文件:{fpath}]")
        text = u._CQ_FILE_RE.sub(u._cq_file_text, text)
        text = u._CQ_VIDEO_RE.sub("[视频]", text)
        text = u._CQ_FORWARD_RE.sub(lambda m: f"[合并转发:{m.group(1)}]", text)
        text = u._CQ_JSON_RE.sub("[卡片]", text)
        text = u._CQ_POKE_RE.sub("[戳一戳]", text)
        text = u._CQ_ANY_RE.sub("", text)
        text = re.sub(r"[ \t]+", " ", text).strip()
        text = u._cq_unescape(text)
        return text, media_urls, media_types

    async def _parse_message_array(
        self, segments: List[dict]
    ) -> Tuple[str, List[str], List[str], Optional[str]]:
        """从 OneBot 段数组解析 (text, media_urls, media_types, reply_id)。

        OneBot 11 事件的 message 字段是段数组
        [{"type": "image", "data": {"file": ..., "url": ...}}, ...]。
        结构化解析比 CQ 字符串正则更可靠（图片 url/file 天然可取）。
        第四项为被引用消息的 message_id（reply 段），供调 get_msg 取原图。
        """
        text_parts: List[str] = []
        media_urls: List[str] = []
        media_types: List[str] = []
        reply_id: Optional[str] = None
        for seg in segments or []:
            if not isinstance(seg, dict):
                continue
            seg_type = seg.get("type", "")
            data = seg.get("data") or {}
            if seg_type == "text":
                text_parts.append(data.get("text", ""))
            elif seg_type == "image":
                try:
                    path = await self._resolve_image(
                        data.get("url", ""), data.get("file", "")
                    )
                    if path:
                        media_urls.append(path)
                        media_types.append("image")
                except Exception as e:
                    logger.debug("[onebot] image resolve failed: %s", e)
                text_parts.append("[图片]")
            elif seg_type == "record":
                try:
                    r_url = data.get("url", "") or ""
                    if not r_url and data.get("file"):
                        rdata = await self._call_action(
                            "get_record",
                            {"file": data["file"], "out_format": "wav"},
                            timeout=15.0,
                        )
                        r_url = rdata.get("url") or rdata.get("file") or ""
                    path = await self._download_audio(r_url)
                    if path:
                        media_urls.append(path)
                        media_types.append("audio/wav")
                except Exception as e:
                    logger.debug("[onebot] voice download failed: %s", e)
                text_parts.append("[语音]")
            elif seg_type == "video":
                try:
                    path = await self._download_media(
                        data.get("url", "") or data.get("file", ""), "video"
                    )
                    if path:
                        media_urls.append(path)
                        media_types.append("video/mp4")
                except Exception as e:
                    logger.debug("[onebot] video download failed: %s", e)
                text_parts.append("[视频]")
            elif seg_type == "file":
                fname = data.get("name") or data.get("file") or "文件"
                try:
                    fpath = await self._resolve_file(data)
                except Exception as e:
                    logger.debug("[onebot] file resolve failed: %s", e)
                    fpath = None
                if fpath:
                    text_parts.append(f"[文件:{fpath}]")
                else:
                    text_parts.append(f"[文件:{fname}]")
            elif seg_type == "face":
                text_parts.append(_load_onebot_utils()._FACE_EMOJI.get(str(data.get("id", "")), "[表情]"))
            elif seg_type == "at":
                qq = str(data.get("qq", ""))
                text_parts.append("@" + ("全体成员" if qq == "all" else qq))
            elif seg_type == "reply":
                reply_id = str(data.get("id", "") or "")
                # 用户偏好不显示引用文本，仅记录 id 供取原图
            elif seg_type == "json":
                text_parts.append("[卡片]")
            elif seg_type == "poke":
                text_parts.append("[戳一戳]")
            # 未知段类型: 忽略
        text = "".join(text_parts)
        text = re.sub(r"[ \t]+", " ", text).strip()
        return text, media_urls, media_types, reply_id

    async def _resolve_image(self, url: str, file: str) -> Optional[str]:
        """把 CQ:image 的 url/file 解析为可读的本地图片路径。

        - url 非空: 直接下载
        - file=base64://...: 直接落盘
        - file=file://...: 本地路径直接用
        - file 是 hash: 调 OneBot get_image API 换取真实 url 再下载
        """
        if url:
            return await self._download_image(url)
        if not file:
            return None
        if file.startswith("base64://"):
            try:
                data = base64.b64decode(file[len("base64://"):])
            except Exception:
                return None
            if len(data) > IMAGE_MAX_BYTES:
                return None
            tmp = Path(tempfile.gettempdir()) / "hermes_onebot"
            tmp.mkdir(exist_ok=True)
            path = tmp / f"img_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.jpg"
            path.write_bytes(data)
            return str(path)
        if file.startswith("file://"):
            p = Path(file[len("file://"):])
            return str(p) if p.exists() else None
        # hash → get_image API 换取真实 URL
        try:
            data = await self._call_action("get_image", {"file": file}, timeout=10.0)
            real = data.get("url") or data.get("file", "")
            if real.startswith(("http://", "https://")):
                return await self._download_image(real)
            if real and not real.startswith(("base64://", "file://")):
                p = Path(real)
                return str(p) if p.exists() else None
        except Exception as e:
            logger.info("[onebot] get_image failed for file=%s: %s", file, e)
        return None

    def _cleanup_tmp_files(self, max_age: float = 6 * 3600) -> None:
        """删除 /tmp/hermes_onebot/ 下超过 max_age 秒的临时媒体文件。

        入站图片/视频下载后只写不删，长期运行会堆积磁盘；每次入站
        媒体处理后顺带清理一次。语音 .wav/.bin 由调用方自行删除，
        这里同样兜底。
        """
        try:
            tmp = Path(tempfile.gettempdir()) / "hermes_onebot"
            if not tmp.is_dir():
                return
            now = time.time()
            for p in tmp.iterdir():
                try:
                    if p.is_file() and (now - p.stat().st_mtime) > max_age:
                        p.unlink(missing_ok=True)
                except (OSError, FileNotFoundError):
                    pass
        except Exception:
            pass

    async def _download_image(self, url: str) -> Optional[str]:
        if not url or url.lower().startswith("base64://"):
            return None
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        timeout = aiohttp.ClientTimeout(total=20.0)
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.read()
        if len(data) > IMAGE_MAX_BYTES:
            logger.debug("[onebot] image too large, skipping (%d bytes)", len(data))
            return None
        ext = mimetypes.guess_extension(resp.headers.get("Content-Type", "")) or ".jpg"
        if ext == ".jpe":
            ext = ".jpg"
        tmp = Path(tempfile.gettempdir()) / "hermes_onebot"
        tmp.mkdir(exist_ok=True)
        path = tmp / f"img_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}{ext}"
        path.write_bytes(data)
        # Shrink oversized images BEFORE the LLM sees them — high-res QQ
        # photos make vision calls slow or time out entirely.
        shrunk = await asyncio.to_thread(self._shrink_image, path)
        return shrunk or str(path)

    async def _download_media(self, url: str, kind: str) -> Optional[str]:
        """通用媒体下载（video/audio/file），带大小上限。

        url 支持 http(s):// 与 base64://；返回本地文件路径。
        kind 仅用于文件名前缀与大小上限选择。
        """
        if not url:
            return None
        max_bytes = MEDIA_MAX_BYTES
        resp = None
        if url.lower().startswith("base64://"):
            try:
                data = base64.b64decode(url[len("base64://"):])
            except Exception:
                return None
        else:
            headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
            timeout = aiohttp.ClientTimeout(total=30.0)
            async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.read()
        if len(data) > max_bytes:
            logger.debug("[onebot] %s too large, skipping (%d bytes)", kind, len(data))
            return None
        ext = mimetypes.guess_extension(resp.headers.get("Content-Type", "")) if resp else ""
        if not ext or ext == ".jpe":
            ext = ".mp4" if kind == "video" else ".bin"
        tmp = Path(tempfile.gettempdir()) / "hermes_onebot"
        tmp.mkdir(exist_ok=True)
        path = tmp / f"{kind}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}{ext}"
        path.write_bytes(data)
        return str(path)

    async def _resolve_file(self, data: dict) -> Optional[str]:
        """入站文件双通道获取（dsh-onebot 回移，2026-08-25）。

        NapCat 容器内的 get_file 路径对宿主不可达，所以：
        1. 优先私聊 CDN 直链 get_private_file_url(file_id) → HTTP 下载；
        2. 回退 get_file 的 base64 / http(s) url 载荷（NapCat 文件服务开启时）。
        成功返回 /tmp/hermes_onebot/ 下的本地路径，失败返回 None。
        """
        fid = data.get("file_id") or data.get("file") or ""
        if not fid:
            return None
        safe_name = (data.get("name") or data.get("file") or "file")[:80] or "file"
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", safe_name) or "file"
        max_bytes = self._max_inbound_file_bytes
        # 1. 私聊直链（不依赖容器内路径）
        try:
            direct = await self._call_action(
                "get_private_file_url", {"file_id": fid}, timeout=15.0
            )
            url = direct.get("url") or ""
            if url:
                path = await self._download_file_bytes(url, safe_name, max_bytes)
                if path:
                    logger.info("[onebot] file fetched via direct link: %s", path)
                    return path
        except Exception as e:
            logger.debug(
                "[onebot] get_private_file_url failed (falling back to get_file): %s", e
            )
        # 2. get_file：base64 或 http(s) url；容器内路径不可达，忽略
        try:
            gdata = await self._call_action("get_file", {"file": fid}, timeout=20.0)
            try:
                size = int(gdata.get("file_size") or 0)
            except (TypeError, ValueError):
                size = 0
            if max_bytes > 0 and size > max_bytes:
                logger.warning("[onebot] inbound file too large (%d B), skipping", size)
                return None
            if gdata.get("base64"):
                path = await self._download_file_bytes(
                    "base64://" + gdata["base64"], safe_name, max_bytes
                )
                if path:
                    logger.info("[onebot] file fetched via get_file base64: %s", path)
                    return path
            url = gdata.get("url") or ""
            if url.startswith(("http://", "https://")):
                path = await self._download_file_bytes(url, safe_name, max_bytes)
                if path:
                    logger.info("[onebot] file fetched via get_file url: %s", path)
                    return path
        except Exception as e:
            logger.debug("[onebot] get_file base64/url path failed: %s", e)
        logger.warning("[onebot] inbound file fetch failed for %s", fid)
        return None

    async def _download_file_bytes(
        self, url: str, safe_name: str, max_bytes: int
    ) -> Optional[str]:
        """下载入站文件到 /tmp/hermes_onebot/<safe_name>，带大小上限。"""
        try:
            if url.lower().startswith("base64://"):
                try:
                    data = base64.b64decode(url[len("base64://"):])
                except Exception:
                    return None
            else:
                headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
                timeout = aiohttp.ClientTimeout(total=30.0)
                async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            return None
                        data = await resp.read()
            if not data:
                return None
            if max_bytes > 0 and len(data) > max_bytes:
                logger.warning("[onebot] inbound file too large (%d B), skipping", len(data))
                return None
            tmp = Path(tempfile.gettempdir()) / "hermes_onebot"
            tmp.mkdir(exist_ok=True)
            path = tmp / safe_name
            if path.exists():
                stem = Path(safe_name).stem
                suffix = Path(safe_name).suffix
                path = tmp / f"{stem}_{int(time.time() * 1000)}{suffix}"
            path.write_bytes(data)
            return str(path)
        except Exception as e:
            logger.warning("[onebot] inbound file download failed: %s", e)
            return None

    def _shrink_image(self, path: Path) -> Optional[str]:
        """Downscale an image to ≤ `image_max_size` px on its long edge.

        Returns the new path when the image was resized, None when it was
        already small enough (or processing failed — the caller keeps the
        original). Animated GIFs collapse to their first frame, which is
        fine for vision analysis.
        """
        max_size = self._image_max_size
        if max_size <= 0:
            return None
        try:
            from PIL import Image

            img = Image.open(path)
            img.load()
        except Exception as e:
            logger.debug("[onebot] image open failed, keeping original: %s", e)
            return None
        try:
            if max(img.size) <= max_size:
                return None
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            out = path.with_suffix(".png" if img.mode == "RGBA" else ".jpg")
            if img.mode == "RGBA":
                img.save(out, format="PNG", optimize=True)
            else:
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                img.save(out, format="JPEG", quality=85)
            return str(out)
        except Exception as e:
            logger.debug("[onebot] image shrink failed, keeping original: %s", e)
            return None

    async def _download_audio(self, url: str) -> Optional[str]:
        """Download a voice clip and convert it to 16 kHz mono WAV.

        QQ voice messages are silk/amr — the gateway STT pipeline expects
        a standard audio file, so ffmpeg converts it (best effort; returns
        None on any failure and the caller degrades to a [语音] marker).
        Accepts http(s) URL or base64:// payload (segment-array file field).
        """
        if not url:
            return None
        if url.lower().startswith("base64://"):
            try:
                data = base64.b64decode(url[len("base64://"):])
            except Exception:
                return None
        else:
            headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
            timeout = aiohttp.ClientTimeout(total=25.0)
            async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.read()
        if len(data) > AUDIO_MAX_BYTES:
            logger.debug("[onebot] voice too large, skipping (%d bytes)", len(data))
            return None

        tmp = Path(tempfile.gettempdir()) / "hermes_onebot"
        tmp.mkdir(exist_ok=True)
        stem = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        in_path = tmp / f"voice_{stem}.bin"
        out_path = tmp / f"voice_{stem}.wav"
        in_path.write_bytes(data)
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-y",
                "-i",
                str(in_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "wav",
                str(out_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                rc = await asyncio.wait_for(proc.wait(), timeout=20.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return None
            if rc != 0 or not out_path.exists():
                logger.debug("[onebot] ffmpeg voice conversion failed (rc=%s)", rc)
                return None
            return str(out_path)
        except (OSError, FileNotFoundError) as e:
            logger.debug("[onebot] ffmpeg unavailable: %s", e)
            return None
        finally:
            in_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if self._ws is None:
            return SendResult(
                success=False,
                error="OneBot WebSocket not connected",
                retryable=True,
            )
        kind, target = _split_chat_id(chat_id)
        # 普通用户会话出站敏感意图审计（软限制兜底观测，非硬拦截）
        if chat_id in getattr(self, "_member_chats", set()):
            hit = _load_onebot_utils().scan_sensitive(content or "")
            if hit:
                logger.warning(
                    "[onebot] restricted-user chat %s reply contains sensitive intent: %r",
                    chat_id, hit,
                )
        try:
            params: Dict[str, Any] = {}
            if kind == "group":
                params["group_id"] = int(target)
            else:
                params["user_id"] = int(target)

            # ── loop 消息合并 ──────────────────────────────────────────
            # gateway 会在一次回复中多次调 send()：中间评论（interim，
            # metadata 带 expect_edits）先发；最终回复（final，带 notify）
            # 后发。为省空间：interim 文本缓冲起来，final 时先合并转发
            # 并撤回那些单独消息，再发最终内容（含 t2i 图片）。
            meta = metadata or {}
            # interim: gateway 中间评论（_send_commentary 带 _interim_send=True；
            # main 于 2026-08-16 独立实现了该标记，旧 key `interim` 一并兼容）
            # final: 流式最终回复带 notify=True；非流式最终回复无标记——
            # 无标记即视为最终（interim 都有标记，无标记=回复周期收尾）
            is_interim = (
                bool(meta.get("interim"))
                or bool(meta.get("_interim_send"))
                or bool(meta.get("expect_edits"))
            )
            is_final = bool(meta.get("notify")) or not is_interim
            # /mode instant（#6 回移）：per-chat 关闭 loop 合并（interim 逐条即时）
            _loop_enabled = self._chat_interim_overrides.get(chat_id, True)
            logger.info(
                "[onebot] send: chat=%s len=%d final=%s interim=%s meta_keys=%s buf=%d",
                chat_id, len(content or ""), is_final, is_interim,
                sorted(meta.keys()), len(self._loop_buffer.get(chat_id, [])),
            )
            # 最终消息：先合并转发（过程回顾先发出），撤回留到内容发送后
            # 顺序：合并转发 → 最终内容（t2i）→ 撤回原 interim（2026-08-14）
            if is_final and _loop_enabled:
                await self._merge_loop_buffer(chat_id, params, kind)
            # 消息发送成功后, interim 纯文本进缓冲（图片/语音等中间媒体不进）
            _buf_sent_ids: List[Tuple[str, str]] = []
            _buf_sent_flag = (
                _loop_enabled
                and is_interim
                and not (metadata or {}).get("media_files")
            )

            # QQ 合并转发指令: [[qq_forward]]名字\n内容\n---\n名字\n内容[[/qq_forward]]
            # 仅群聊支持 send_forward_msg; 私聊忽略该标记走普通文本。
            raw_content = content or ""
            fwd_match = _load_onebot_utils()._FORWARD_RE.search(raw_content)
            if fwd_match and kind == "group":
                nodes = self._parse_forward_blocks(fwd_match.group(1))
                if nodes:
                    try:
                        await self._call_action(
                            "send_forward_msg",
                            {"group_id": int(target), "messages": nodes},
                            timeout=30.0,
                        )
                    except Exception as e:
                        logger.warning("[onebot] send_forward_msg failed: %s", e)
                raw_content = _load_onebot_utils()._FORWARD_RE.sub("", raw_content).strip()
            content = raw_content

            # No [CQ:reply] prefix — user prefers plain replies without a
            # quoted reference to the triggering message.
            media_segments: List[Dict[str, Any]] = []
            if metadata:
                media_files = metadata.get("media_files") or []
                for f in media_files:
                    # Hermes media_files entries are (path, is_voice) tuples
                    # (extract_media contract) — unpack defensively so a tuple
                    # never reaches Path() as a TypeError and silently drops.
                    media_path = f[0] if isinstance(f, (tuple, list)) else f
                    b64 = await self._file_to_base64(media_path)
                    if b64:
                        media_segments.append(
                            {"type": "image", "data": {"file": f"base64://{b64}"}}
                        )

            # QQ doesn't render Markdown — convert common syntax to readable
            # plain text BEFORE splitting so text chunks are clean.
            raw_content = content or ""
            content = _load_onebot_utils().strip_markdown(raw_content)

            parts = _load_onebot_utils()._split_reply(content or "", self._split_length)

            # Long content → single text-image message instead of text.
            # Text-image path receives the RAW markdown so the AstrBot-style
            # renderer can draw bold/headers/tables etc. properly.
            if (
                self._text_image_threshold > 0
                and len(raw_content) > self._text_image_threshold
            ):
                try:
                    title = None
                    nick = self._nicknames.get(chat_id, "")
                    if nick:
                        title = f"To {nick}"
                    png_bytes = await asyncio.to_thread(
                        render_text_image, raw_content, title
                    )
                    b64 = base64.b64encode(png_bytes).decode("ascii")
                    image_params = dict(params)
                    image_params["message"] = [
                        {"type": "image", "data": {"file": f"base64://{b64}"}}
                    ] + media_segments
                    data = await self._call_action("send_msg", image_params)
                    mid = data.get("message_id")
                    if _buf_sent_flag and mid is not None:
                        _buf_sent_ids.append((str(mid), raw_content))
                    # t2i 结果发送完成后撤回原 interim（合并转发已在内容前发出）
                    if is_final:
                        await self._recall_loop_buffer(chat_id)
                    return SendResult(
                        success=True, message_id=str(mid) if mid is not None else None
                    )
                except Exception as e:
                    logger.warning(
                        "[onebot] text-image render failed (%s) — falling back to text chunks", e
                    )

            if not parts and not media_segments:
                if is_final:
                    await self._recall_loop_buffer(chat_id)
                return SendResult(success=True, message_id=None)

            last_message_id: Optional[str] = None
            for idx, part in enumerate(parts):
                chunk_segments: List[Dict[str, Any]] = [
                    {"type": "text", "data": {"text": part}}
                ]
                # Attach media to the final chunk.
                if idx == len(parts) - 1:
                    chunk_segments.extend(media_segments)
                chunk_params = dict(params)
                chunk_params["message"] = chunk_segments
                data = await self._call_action("send_msg", chunk_params)
                mid = data.get("message_id")
                if mid is not None:
                    last_message_id = str(mid)
                    if _buf_sent_flag:
                        _buf_sent_ids.append((last_message_id, part))

            # interim 文本消息记入缓冲, 等 final 到达后合并转发
            if _buf_sent_ids:
                self._loop_buffer.setdefault(chat_id, []).extend(_buf_sent_ids)
                self._loop_buffer_ts[chat_id] = time.time()
                # #2 回移：每条 interim 独立计时，90s 内 final 未结算则单独撤回
                for mid, text in _buf_sent_ids:
                    asyncio.create_task(self._auto_recall_interim(chat_id, mid, text))
            else:
                # 没有新 interim 写入时顺带做超时兜底：interim 后 5 分钟内
                # final 未到（gateway 中断/异常），清掉残留缓冲防滞留
                ts = self._loop_buffer_ts.get(chat_id)
                if ts and (time.time() - ts) > 300 and self._loop_buffer.get(chat_id):
                    logger.info("[onebot] loop buffer expired for %s, dropping %d item(s)",
                                chat_id, len(self._loop_buffer.get(chat_id, [])))
                    self._loop_buffer.pop(chat_id, None)
                    self._loop_buffer_ts.pop(chat_id, None)

            # 最终内容已发完，撤回原 interim（合并转发已在内容前发出）
            if is_final:
                await self._recall_loop_buffer(chat_id)
            return SendResult(success=True, message_id=last_message_id)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("[onebot] send failed: %s", e)
            return SendResult(success=False, error=str(e), retryable=True)

    async def _merge_loop_buffer(
        self, chat_id: str, params: Dict[str, Any], kind: str
    ) -> None:
        """把一次回复周期内缓冲的 interim 消息结算为一条（先发）。

        顺序（2026-08-14）：结算卡 → 最终内容 → 撤回原消息。
        结算形态（#3 回移，2026-08-25）优先回合末 t2i 小结卡（渲染整轮
        interim 为一张文字图卡片）；渲染/发送失败回退合并转发。

        - 仅当缓冲 ≥2 条时才结算（单条不值得；转发本身也占空间）
        - 结算成功后待撤回 id 记入 _pending_recalls；失败则保留原消息（不丢内容）
        - 群聊用 send_forward_msg(group_id)，私聊用 send_private_forward_msg(user_id)
        """
        buf = self._loop_buffer.pop(chat_id, None)
        self._loop_buffer_ts.pop(chat_id, None)
        if not buf or len(buf) < 2:
            return
        # 1) 回合末小结卡：渲染整轮 interim 为一张 t2i 卡片
        try:
            summary_text = "\n\n".join(
                _load_onebot_utils().strip_markdown(t)[:300] for _, t in buf
            )
            if summary_text.strip():
                png = await asyncio.to_thread(render_text_image, summary_text, "本轮进展")
                if png:
                    tmp = Path(tempfile.gettempdir()) / "hermes_onebot"
                    tmp.mkdir(exist_ok=True)
                    card = tmp / f"loop_summary_{int(time.time() * 1000)}.png"
                    card.write_bytes(png)
                    res = await self.send_image_file(chat_id, str(card))
                    if getattr(res, "success", False):
                        self._pending_recalls[chat_id] = [mid for mid, _ in buf]
                        return
                    logger.info(
                        "[onebot] loop summary card send failed, falling back to merge-forward"
                    )
        except Exception as e:
            logger.info(
                "[onebot] loop summary render failed, falling back to merge-forward: %s", e
            )
        # 4) 回退：合并转发
        try:
            uin = str(self._self_id or self._bot_qq or "0")
            nodes = []
            for mid, text in buf:
                content = _load_onebot_utils().strip_markdown(text)[:500]
                if not content.strip():
                    content = "(中间消息)"
                nodes.append(
                    {
                        "type": "node",
                        "data": {
                            "uin": uin,
                            "name": "Hermes",
                            "content": [{"type": "text", "data": {"text": content}}],
                        },
                    }
                )
            if not nodes:
                return
            action = "send_forward_msg" if kind == "group" else "send_private_forward_msg"
            fwd_params = dict(params)
            fwd_params["messages"] = nodes
            await self._call_action(action, fwd_params, timeout=30.0)
            # 转发成功后才登记撤回（撤回在最终内容发送后执行）
            self._pending_recalls[chat_id] = [mid for mid, _ in buf]
        except Exception as e:
            self._pending_recalls.pop(chat_id, None)
            logger.info("[onebot] loop merge failed, keeping original messages: %s", e)

    async def _auto_recall_interim(self, chat_id: str, mid: str, text: str) -> None:
        """#2 回移：单条 interim 超时自动撤回（dsh 语义）。

        interim 发送后计时 `_interim_recall_seconds`（默认 90s）；到点时若该
        条仍在缓冲（final 尚未结算），单独撤回并从缓冲移除。final 结算或
        新用户消息清缓冲后，任务到点发现条目已不在 → 无事（自愈，无需取消）。
        """
        delay = self._interim_recall_seconds
        if delay <= 0:
            return
        try:
            await asyncio.sleep(delay)
            if self._stopping:
                return
            buf = self._loop_buffer.get(chat_id)
            if not buf or (mid, text) not in buf:
                return
            buf.remove((mid, text))
            logger.info("[onebot] auto-recall interim %s after %ss (no final)", mid, delay)
            try:
                await self._call_action("delete_msg", {"message_id": mid}, timeout=10.0)
            except Exception as e:
                logger.debug("[onebot] auto-recall delete_msg failed for %s: %s", mid, e)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("[onebot] auto-recall interim failed: %s", e)

    async def _recall_loop_buffer(self, chat_id: str) -> None:
        """撤回已合并转发的原 interim 消息（在最终内容发送完成后调用）。

        #4 回移：批量 delete_msg 每条间隔 60ms（RECALL_SPACING_MS），突发
        连续撤回会撞 NapCat 接口超时。
        """
        mids = self._pending_recalls.pop(chat_id, None)
        if not mids:
            return
        for mid in mids:
            try:
                await self._call_action(
                    "delete_msg", {"message_id": mid}, timeout=10.0
                )
            except Exception as e:
                logger.debug("[onebot] delete_msg failed for %s: %s", mid, e)
            await asyncio.sleep(0.06)  # 60ms 撤回间隔，防 NapCat 限速

    async def _file_to_base64(self, path: str, max_bytes: int = IMAGE_MAX_BYTES) -> Optional[str]:
        try:
            p = Path(path)
            if not p.exists() or p.stat().st_size > max_bytes:
                return None
            data = await asyncio.to_thread(p.read_bytes)
            return base64.b64encode(data).decode("ascii")
        except Exception as e:
            logger.debug("[onebot] media read failed: %s", e)
            return None

    async def _call_action(self, action: str, params: Dict[str, Any], timeout: float = ACTION_TIMEOUT) -> Dict[str, Any]:
        ws = self._ws
        if ws is None:
            raise ConnectionError("OneBot WebSocket not connected")
        echo = str(uuid.uuid4())
        fut: "asyncio.Future[Dict[str, Any]]" = asyncio.get_running_loop().create_future()
        self._pending_actions[echo] = fut
        try:
            await ws.send_str(json.dumps({"action": action, "params": params, "echo": echo}))
            resp = await asyncio.wait_for(fut, timeout)
        finally:
            self._pending_actions.pop(echo, None)
        if resp.get("status") != "ok":
            raise RuntimeError(
                f"OneBot action {action} failed: {resp.get('wording') or resp.get('retcode')}"
            )
        return resp.get("data") or {}

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        kind, target = _split_chat_id(chat_id)
        return {"name": chat_id, "type": "group" if kind == "group" else "dm"}

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Show the QQ "typing…" bubble — NapCat `set_input_status`.

        QQ only surfaces input status for C2C (private) chats; group
        chats have no typing indicator, so those are skipped. Failures
        are silently ignored (the gateway bounds this call itself).
        """
        if self._ws is None:
            return
        kind, target = _split_chat_id(chat_id)
        if kind != "private":
            return
        try:
            await self._call_action(
                "set_input_status",
                {"user_id": target, "event_type": 1},
                timeout=3.0,
            )
        except Exception as e:
            logger.debug("[onebot] send_typing failed: %s", e)

    async def stop_typing(self, chat_id: str) -> None:
        """Clear the QQ input-status bubble (private chats only)."""
        if self._ws is None:
            return
        kind, target = _split_chat_id(chat_id)
        if kind != "private":
            return
        try:
            await self._call_action(
                "set_input_status",
                {"user_id": target, "event_type": 0},
                timeout=3.0,
            )
        except Exception as e:
            logger.debug("[onebot] stop_typing failed: %s", e)

    # ------------------------------------------------------------------
    # Rich media delivery (OneBot segments)
    # ------------------------------------------------------------------
    def _parse_forward_blocks(self, inner: str) -> Optional[List[Dict[str, Any]]]:
        """解析 [[qq_forward]] 内容为 OneBot node 数组。

        块格式: 第一行是转发者名字, 其余为内容; 块之间用 --- 分隔。
        """
        uin = str(self._self_id or self._bot_qq or "0")
        nodes: List[Dict[str, Any]] = []
        for block in inner.split("\n---\n"):
            lines = [l.rstrip() for l in block.split("\n") if l.strip()]
            if not lines:
                continue
            name = lines[0][:24]
            text = "\n".join(lines[1:]).strip()
            if not text:
                continue
            nodes.append(
                {
                    "type": "node",
                    "data": {
                        "uin": uin,
                        "name": name,
                        "content": [
                            {"type": "text", "data": {"text": text[:500]}}
                        ],
                    },
                }
            )
        return nodes or None

    async def _send_media(
        self,
        chat_id: str,
        segments: List[Dict[str, Any]],
        reply_to: Optional[str] = None,
        caption: Optional[str] = None,
    ) -> SendResult:
        kind, target = _split_chat_id(chat_id)
        params: Dict[str, Any] = {}
        try:
            if kind == "group":
                params["group_id"] = int(target)
            else:
                params["user_id"] = int(target)
        except (ValueError, TypeError):
            logger.warning("[onebot] bad chat_id for media send: %r", chat_id)
            return SendResult(success=False, error=f"bad chat_id: {chat_id}", retryable=False)
        msg = list(segments)
        if caption:
            msg.insert(0, {"type": "text", "data": {"text": caption}})
        params["message"] = msg
        try:
            data = await self._call_action("send_msg", params)
            mid = data.get("message_id")
            return SendResult(
                success=True, message_id=str(mid) if mid is not None else None
            )
        except Exception as e:
            logger.warning("[onebot] media send failed: %s", e)
            return SendResult(success=False, error=str(e), retryable=True)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """URL 图片直发: image segment 的 url 字段, NapCat 自行下载."""
        return await self._send_media(
            chat_id,
            [{"type": "image", "data": {"url": image_url}}],
            reply_to,
            caption,
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
        b64 = await self._file_to_base64(str(image_path))
        if not b64:
            return SendResult(success=False, error="image too large or unreadable", retryable=False)
        return await self._send_media(
            chat_id,
            [{"type": "image", "data": {"file": f"base64://{b64}"}}],
            reply_to,
            caption,
        )

    async def send_multiple_images(
        self,
        chat_id: str,
        images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0,
    ) -> None:
        """批量图片: file:// → base64, http(s):// → URL 直发; 一条消息最多 9 图."""
        from urllib.parse import unquote

        segs: List[Dict[str, Any]] = []
        for uri, _alt in images:
            if human_delay > 0:
                await asyncio.sleep(human_delay)
            if uri.startswith("file://"):
                path = unquote(uri[7:])
                b64 = await self._file_to_base64(path)
                if b64:
                    segs.append({"type": "image", "data": {"file": f"base64://{b64}"}})
            elif uri.startswith(("http://", "https://")):
                segs.append({"type": "image", "data": {"url": uri}})
        for i in range(0, len(segs), 9):
            batch = segs[i : i + 9]
            if batch:
                await self._send_media(chat_id, batch)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        b64 = await self._file_to_base64(str(audio_path), max_bytes=MEDIA_MAX_BYTES)
        if not b64:
            return SendResult(success=False, error="voice too large or unreadable", retryable=False)
        return await self._send_media(
            chat_id,
            [{"type": "record", "data": {"file": f"base64://{b64}"}}],
            reply_to,
            caption,
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
        b64 = await self._file_to_base64(str(video_path), max_bytes=MEDIA_MAX_BYTES)
        if not b64:
            return SendResult(success=False, error="video too large or unreadable", retryable=False)
        return await self._send_media(
            chat_id,
            [{"type": "video", "data": {"file": f"base64://{b64}"}}],
            reply_to,
            caption,
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
        b64 = await self._file_to_base64(str(file_path), max_bytes=MEDIA_MAX_BYTES)
        if not b64:
            return SendResult(success=False, error="file too large or unreadable", retryable=False)
        name = file_name or Path(str(file_path)).name
        return await self._send_media(
            chat_id,
            [{"type": "file", "data": {"file": f"base64://{b64}", "name": name}}],
            reply_to,
            caption,
        )


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[Any]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Out-of-process cron delivery via the OneBot HTTP API.

    Used when cron runs in a separate process from the gateway (no live
    adapter available). Text goes out as one message; each media file is
    sent as a record/image/file CQ segment. ``voice_mount``
    ({host, container}) maps host audio paths so NapCat inside Docker can
    read them; paths are normalized to forward slashes before comparison so
    Windows-style backslashes don't silently fail the prefix check.
    """
    extra = getattr(pconfig, "extra", {}) or {}
    http_url = os.getenv("ONEBOT_HTTP_URL") or extra.get("http_url", "")
    token = os.getenv("ONEBOT_ACCESS_TOKEN") or extra.get("access_token", "")
    voice_mount = extra.get("voice_mount") or {}
    # Fallback: pconfig.extra may be stale (cached platform config); re-read
    # voice_mount from config.yaml so host→container mapping stays live.
    if not voice_mount:
        try:
            from gateway.config import load_gateway_config

            _cfg = load_gateway_config()
            _pc = _cfg.platforms.get(Platform("onebot"))
            if _pc is not None:
                voice_mount = (getattr(_pc, "extra", {}) or {}).get("voice_mount") or {}
            else:
                logger.warning("OneBot standalone: onebot platform not in config")
        except Exception as e:
            logger.warning("OneBot standalone: voice_mount fallback failed: %s", e)
    if not http_url:
        return {"error": "OneBot standalone send: ONEBOT_HTTP_URL not configured"}

    # Resolve target: chat_id is the canonical "private:<id>" / "group:<id>"
    # form (cron ONEBOT_HOME_CHANNEL), but bare numeric ids are tolerated.
    raw_target = str(chat_id)
    api, key = "send_private_msg", "user_id"
    if ":" in raw_target:
        kind, target = raw_target.split(":", 1)
        if kind == "group":
            api, key = "send_group_msg", "group_id"
    else:
        target = raw_target
    try:
        target_int = int(target)
    except (TypeError, ValueError):
        return {"error": f"OneBot standalone send: bad chat_id {chat_id!r}"}

    try:
        import aiohttp

        async def _post(action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            headers = {"Content-Type": "application/json"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{http_url.rstrip('/')}/{action}",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    data = await resp.json()
                    if data.get("retcode") != 0:
                        raise RuntimeError(f"OneBot API {action} failed: {data}")
                    return data

        # 1. Text message
        if message and message.strip():
            await _post(api, {key: target_int, "message": message})

        # 2. Media files (voice / image / document) — entries are
        # (path, is_voice) tuples per the Hermes media_files contract.
        sent_any_media = False
        for mpath in media_files or []:
            if isinstance(mpath, (tuple, list)):
                p = str(mpath[0]) if len(mpath) > 0 else ""
                is_voice = bool(mpath[1]) if len(mpath) > 1 else False
            else:
                p = str(mpath)
                is_voice = False
            if not p:
                continue
            ext = Path(p).suffix.lower() if p else ""
            # voice_mount: host path → container path so NapCat in Docker can
            # see the file. Normalize separators before comparing (Windows
            # backslash vs config forward slash).
            target = p
            host_prefix = (voice_mount or {}).get("host", "") or ""
            cont_prefix = (voice_mount or {}).get("container", "") or ""
            norm_p = p.replace("\\", "/")
            norm_host = host_prefix.replace("\\", "/") if host_prefix else ""
            if norm_host and norm_p.startswith(norm_host):
                target = cont_prefix + norm_p[len(norm_host):]
            if is_voice or ext in _AUDIO_EXTS:
                cq = f"[CQ:record,file={target}]"
            elif ext in _IMAGE_EXTS:
                cq = f"[CQ:image,file={target}]"
            else:
                cq = f"[CQ:file,file={target}]"
            await _post(api, {key: target_int, "message": cq})
            sent_any_media = True

        return {"success": True, "message_id": str(int(time.time() * 1000))}
    except Exception as e:
        return {"error": f"OneBot standalone send failed: {e}"}


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def check_requirements() -> bool:
    return AIOHTTP_AVAILABLE


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    mode = str(extra.get("mode", "reverse")).strip().lower()
    if mode not in ("reverse", "forward"):
        return False
    return True


def _build_adapter(config):
    return OneBotAdapter(config)


def _is_connected(config) -> bool:
    # Best-effort: connected when config is present; real state lives on the
    # live adapter (checked by the gateway runner).
    return validate_config(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="onebot",
        label="QQ (OneBot)",
        adapter_factory=_build_adapter,
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=_is_connected,
        install_hint="OneBot needs a running NapCat / Lagrange / LLOneBot instance",
        allowed_users_env="ONEBOT_ALLOWED_USERS",
        allow_all_env="ONEBOT_ALLOW_ALL_USERS",
        cron_deliver_env_var="ONEBOT_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🐧",
        platform_hint=(
            "You are chatting via QQ (OneBot). Plain text only — no markdown. "
            "Group chats: users @you or reply to you. Keep replies concise. "
            "Group messages from restricted users carry a [受限用户:仅问答] prefix: "
            "only answer quick questions / public info / image analysis / group "
            "summaries for them. NEVER execute file operations, terminal commands, "
            "config changes, service restarts, Home Assistant device control, "
            "cross-platform sends, or cron operations for restricted users — "
            "politely refuse and explain no permission."
        ),
    )
    # 模型工具（qq_send_* / qq_napcat_api / qq_group_history）：gateway 进程
    # 内也主动注册（CLI/TUI 由 provides_tools 预加载机制负责）。失败不影响
    # 平台本身启动。
    try:
        from .tools import register_tools

        register_tools(ctx)
    except Exception as e:
        logger.warning("[onebot] tools registration failed: %s", e)

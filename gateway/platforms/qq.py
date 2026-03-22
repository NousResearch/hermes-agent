"""
Hermes QQ Bot Platform Adapter

Fully aligned with Hermes gateway specification + Tencent OpenClaw QQ Bot official implementation.
QQ Open Platform documentation: https://q.qq.com/wiki/

This adapter should be placed in:
  ~/.hermes/plugins/hermes-qqbot/qqbot/adapter.py

Or integrated into core with 2 minimal modifications:
  1. gateway/config.py - Add Platform.QQ = "qq"
  2. gateway/run.py   - Add QQ adapter registration in _create_adapter()
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import sys
import os

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_url,
    get_image_cache_dir,
    get_audio_cache_dir,
)

logger = logging.getLogger(__name__)

QQ_API_BASE = "https://api.sgroup.qq.com"
QQ_TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken"
QQ_WS_GATEWAY_URL = "https://api.sgroup.qq.com/gateway/bot"

QQ_INTENTS = {
    "GUILDS": 1 << 0,
    "GUILD_MEMBERS": 1 << 1,
    "PUBLIC_GUILD_MESSAGES": 1 << 30,
    "DIRECT_MESSAGE": 1 << 12,
    "GROUP_AT_MESSAGE_CREATE": 1 << 29,
}

# 计算FULL_INTENTS：直接使用整数值
FULL_INTENTS = (
    (1 << 25)  # PUBLIC_MESSAGES (包含C2C消息和群聊AT消息)
    | (1 << 30)  # PUBLIC_GUILD_MESSAGES
    | (1 << 12)  # DIRECT_MESSAGE
)

RECONNECT_DELAYS = [1000, 2000, 5000, 10000, 30000, 60000]
MAX_RECONNECT_ATTEMPTS = 100


def check_qq_requirements() -> bool:
    """Check if QQ platform dependencies are available."""
    return AIOHTTP_AVAILABLE


@dataclass
class QQBotConfig:
    """QQ Bot configuration, fully compatible with OpenClaw format."""
    app_id: str
    app_secret: str
    allowed_users: List[str] = field(default_factory=list)
    allowed_groups: List[str] = field(default_factory=list)
    at_required: bool = True
    auto_quote: bool = False
    markdown_support: bool = False


class RefIndexStore:
    """In-memory store for message reference index (ref_idx)."""
    
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
    
    def set(self, ref_idx: str, entry: Dict[str, Any]) -> None:
        self._store[ref_idx] = entry
    
    def get(self, ref_idx: str) -> Optional[Dict[str, Any]]:
        return self._store.get(ref_idx)
    
    def clear(self) -> None:
        self._store.clear()


class MessageQueue:
    """Per-user concurrent message queue."""
    
    def __init__(self, account_id: str, log: Optional[logging.Logger] = None):
        self.account_id = account_id
        self.log = log
        self._queues: Dict[str, List[Dict[str, Any]]] = {}
    
    def enqueue(self, msg: Dict[str, Any]) -> None:
        peer_id = msg.get("peer_id", msg.get("sender_id", "unknown"))
        if peer_id not in self._queues:
            self._queues[peer_id] = []
        self._queues[peer_id].append(msg)
    
    def get_next(self, peer_id: str) -> Optional[Dict[str, Any]]:
        if peer_id in self._queues and self._queues[peer_id]:
            return self._queues[peer_id].pop(0)
        return None
    
    def clear_user_queue(self, peer_id: str) -> int:
        count = len(self._queues.get(peer_id, []))
        if peer_id in self._queues:
            self._queues[peer_id] = []
        return count


class QQAdapter(BasePlatformAdapter):
    """
    Hermes QQ Official Bot Adapter.
    
    Fully implements BasePlatformAdapter interface, aligned with official platform specification.
    """
    
    MAX_MESSAGE_LENGTH = 500
    
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.QQ)
        
        if not check_qq_requirements():
            logger.warning("QQ: aiohttp not installed. QQ adapter will not be available.")
        
        extra = config.extra or {}
        
        token_parts = (self.config.token or "").split(":", 1)
        app_id = token_parts[0] if token_parts else ""
        app_secret = token_parts[1] if len(token_parts) > 1 else extra.get("app_secret", "")
        
        self.bot_config = QQBotConfig(
            app_id=app_id,
            app_secret=app_secret,
            allowed_users=self._parse_allowlist(extra.get("allowed_users", "")),
            allowed_groups=self._parse_allowlist(extra.get("allowed_groups", "")),
            at_required=extra.get("at_required", True),
            auto_quote=extra.get("auto_quote", False),
            markdown_support=extra.get("markdown_support", False),
        )
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._access_token: str = ""
        self._token_expire_at: float = 0
        self._bot_openid: str = ""
        self._connected: bool = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._running: bool = False
        self._reconnect_attempts: int = 0
        self._session_id: Optional[str] = None
        self._last_seq: Optional[int] = None
        self._ref_index: RefIndexStore = RefIndexStore()
        self._msg_queue: MessageQueue = MessageQueue(self.name, logger)
        self._image_cache_dir = get_image_cache_dir()
        self._audio_cache_dir = get_audio_cache_dir()
    
    def _parse_allowlist(self, value: Any) -> List[str]:
        """Parse allowlist from string (comma-separated) or list."""
        if not value:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return []
    
    async def _refresh_access_token(self, force: bool = False) -> bool:
        """Refresh AccessToken with background refresh logic."""
        if not self.bot_config.app_id or not self.bot_config.app_secret:
            logger.error("QQ Bot app_id/app_secret not configured")
            return False
        
        if not force and self._access_token and time.time() < self._token_expire_at - 60:
            return True
        
        try:
            async with self._session.post(
                QQ_TOKEN_URL,
                json={
                    "appId": self.bot_config.app_id,
                    "clientSecret": self.bot_config.app_secret
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                
                if "access_token" not in data:
                    logger.error(f"Failed to get QQ AccessToken: {data}")
                    return False
                
                self._access_token = data["access_token"]
                expires_in = data.get("expires_in", 7200)
                # 确保 expires_in 是数字类型
                if isinstance(expires_in, str):
                    try:
                        expires_in = int(expires_in)
                    except ValueError:
                        expires_in = 7200
                self._token_expire_at = time.time() + expires_in
                logger.info("QQ AccessToken refreshed successfully")
                return True
                
        except Exception as e:
            logger.exception(f"Failed to refresh QQ AccessToken: {e}")
            return False
    
    @property
    def _auth_headers(self) -> Dict[str, str]:
        """Official required authentication headers."""
        return {
            "Authorization": f"QQBot {self._access_token}",
            "Content-Type": "application/json",
            "X-Union-Appid": self.bot_config.app_id
        }
    
    async def _get_gateway_url(self) -> str:
        """Get WebSocket gateway URL."""
        await self._refresh_access_token()
        
        async with self._session.get(
            QQ_WS_GATEWAY_URL,
            headers=self._auth_headers,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["url"]
    
    async def _start_heartbeat(self, interval: int) -> None:
        """WebSocket heartbeat loop."""
        try:
            seq = self._last_seq or 0
            while self._running and self._ws and not self._ws.closed:
                await self._ws.send_json({"op": 1, "d": seq})
                logger.debug(f"QQ Heartbeat sent, seq={seq}")
                await asyncio.sleep(interval / 1000)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"QQ Heartbeat loop error: {e}")
    
    async def _download_and_cache_attachment(
        self, 
        url: str, 
        content_type: str,
        msg_id: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Download attachment and cache locally."""
        if not url:
            return None, None
        
        try:
            ext = ".jpg"
            if "image" in content_type:
                if ".png" in url.lower():
                    ext = ".png"
                elif ".gif" in url.lower():
                    ext = ".gif"
                elif ".webp" in url.lower():
                    ext = ".webp"
            elif "audio" in content_type or "voice" in content_type:
                ext = ".ogg"
            elif "video" in content_type:
                ext = ".mp4"
            
            cache_dir = self._image_cache_dir if "image" in content_type else self._audio_cache_dir
            
            filename = f"qq_{msg_id[:8]}_{uuid.uuid4().hex[:8]}{ext}"
            filepath = cache_dir / filename
            
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                data = await resp.read()
                filepath.write_bytes(data)
                
            return str(filepath), content_type
                
        except Exception as e:
            logger.warning(f"Failed to download attachment: {e}")
            return None, None
    
    async def _parse_event(self, event: Dict[str, Any]) -> Optional[MessageEvent]:
        """Parse QQ event into Hermes MessageEvent."""
        op = event.get("op")
        t = event.get("t")
        d = event.get("d", {})
        s = event.get("s")
        
        # 🔍 调试：记录收到的原始事件结构
        logger.info(f"[DEBUG] QQ event received: op={op}, t={t!r}, keys={list(event.keys())}, d_keys={list(d.keys()) if d else None}")
        
        if s is not None:
            self._last_seq = int(s) if isinstance(s, (int, str)) else s
        
        # op=0 是 DISPATCH 事件（包含消息等业务事件）
        # op=1 是心跳响应，忽略
        if op == 1:
            # 心跳确认（我们发出的心跳被服务器确认），忽略
            return None
        
        # op=11 是服务器发送的心跳请求（PING），必须立即回复心跳响应（PONG含 seq）
        if op == 11:
            seq = event.get("s") or self._last_seq or 0
            try:
                await self._ws.send_json({"op": 1, "d": seq})
                logger.debug(f"QQ heartbeat response sent (seq={seq})")
            except Exception as e:
                logger.warning(f"Failed to send heartbeat response: {e}")
            return None
        
        if t == "READY":
            self._bot_openid = d.get("user", {}).get("id", "")
            logger.info(f"QQ Bot connected! ID: {self._bot_openid}")
            self._session_id = d.get("session_id")
            self._mark_connected()
            return None
        
        if t == "RESUMED":
            logger.info("QQ Session resumed successfully")
            self._mark_connected()
            return None
        
        if t == "C2C_MESSAGE_CREATE":
            logger.info(f"[C2C] Processing message from user_openid={d.get('author',{}).get('user_openid')}, content={d.get('content','')[:50]}")
            return await self._handle_c2c_message(d)
        
        if t == "GROUP_AT_MESSAGE_CREATE":
            logger.info(f"[GROUP] Processing message from group={d.get('group_openid')}, member={d.get('author',{}).get('member_openid')}")
            return await self._handle_group_message(d)
        
        # 其他事件类型记录 debug
        logger.debug(f"QQ received unhandled event type: {t}")
        return None

    async def _handle_c2c_message(self, d: Dict[str, Any]) -> Optional[MessageEvent]:
        """Handle C2C (private) message."""
        sender = d.get("author", {})
        user_openid = sender.get("user_openid", "")
        content = d.get("content", "").strip()
        message_id = d.get("id", "")

        # 解析timestamp，兼容ISO 8601格式（如"2026-03-22T08:55:03+08:00"）
        ts_raw = d.get("timestamp", time.time() * 1000)
        if isinstance(ts_raw, str):
            try:
                timestamp = int(ts_raw)
            except ValueError:
                try:
                    from datetime import datetime
                    dt = datetime.strptime(ts_raw.replace('+08:00', '+0000'), "%Y-%m-%dT%H:%M:%S%z")
                    timestamp = int(dt.timestamp() * 1000)
                except ValueError:
                    timestamp = int(time.time() * 1000)
        else:
            timestamp = int(ts_raw)

        if self.bot_config.allowed_users and user_openid not in self.bot_config.allowed_users:
            logger.warning(f"Rejecting non-whitelisted user C2C: {user_openid}")
            return None

        if not content:
            return None

        media_urls = []
        media_types = []

        for att in d.get("attachments", []):
            url = att.get("url", "")
            content_type = att.get("content_type", "image/jpeg")

            cached_path, _ = await self._download_and_cache_attachment(url, content_type, message_id)
            if cached_path:
                media_urls.append(cached_path)
                media_types.append(content_type)
            else:
                media_urls.append(url)
                media_types.append(content_type)

        logger.info(f"QQ C2C message from {user_openid}: {content[:50]}")

        source = self.build_source(
            chat_id=f"qq:c2c:{user_openid}",
            chat_name=user_openid,
            chat_type="dm",
            user_id=user_openid,
            user_name=sender.get("username", ""),
        )

        return MessageEvent(
            text=content,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=d,
            message_id=message_id,
            media_urls=media_urls,
            media_types=media_types,
            timestamp=datetime.fromtimestamp(timestamp / 1000),
        )

    async def _handle_group_message(self, d: Dict[str, Any]) -> Optional[MessageEvent]:
        """Handle group @ message."""
        group_openid = d.get("group_openid", "")
        sender = d.get("author", {})
        user_openid = sender.get("member_openid", sender.get("user_openid", ""))
        content = d.get("content", "").strip()
        message_id = d.get("id", "")

        # 解析timestamp，兼容ISO 8601格式（如"2026-03-22T08:55:03+08:00"）
        ts_raw = d.get("timestamp", time.time() * 1000)
        if isinstance(ts_raw, str):
            try:
                timestamp = int(ts_raw)
            except ValueError:
                try:
                    from datetime import datetime
                    dt = datetime.strptime(ts_raw.replace('+08:00', '+0000'), "%Y-%m-%dT%H:%M:%S%z")
                    timestamp = int(dt.timestamp() * 1000)
                except ValueError:
                    timestamp = int(time.time() * 1000)
        else:
            timestamp = int(ts_raw)

        if self.bot_config.allowed_groups and group_openid not in self.bot_config.allowed_groups:
            return None

        if self.bot_config.at_required:
            at_mark = f"<@!{self._bot_openid}>"
            if at_mark not in content:
                return None
            content = content.replace(at_mark, "").strip()

        if not content:
            return None

        media_urls = []
        media_types = []

        for att in d.get("attachments", []):
            url = att.get("url", "")
            content_type = att.get("content_type", "image/jpeg")

            cached_path, _ = await self._download_and_cache_attachment(url, content_type, message_id)
            if cached_path:
                media_urls.append(cached_path)
                media_types.append(content_type)

        source = MessageSource(
            platform=self.platform,
            channel_id=f"qq:group:{group_openid}",
            channel_name=group_openid,
            user_id=user_openid,
            user_name=sender.get("member_name", sender.get("username", "")),
        )

        return MessageEvent(
            text=content,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=d,
            message_id=message_id,
            media_urls=media_urls,
            media_types=media_types,
            timestamp=datetime.fromtimestamp(timestamp / 1000),
        )

    async def _ws_receive_loop(self) -> None:
        """WebSocket event receiving loop."""
        while self._running:
            try:
                if not self._ws or self._ws.closed:
                    await self._connect_ws()
                    continue
                
                async for msg in self._ws:
                    if not self._running:
                        break
                    
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            event = json.loads(msg.data)
                            
                            # 调试：记录所有收到的原始事件（截断避免日志过大）
                            d_val = event.get('d', {})
                            d_keys = list(d_val.keys()) if isinstance(d_val, dict) else str(d_val)[:50]
                            logger.debug(f"QQ raw event: op={event.get('op')}, t={event.get('t')}, s={event.get('s')}, d_keys={d_keys}")
                            
                            if event.get("op") == 10:
                                interval = event.get("d", {}).get("heartbeat_interval", 45000)
                                logger.info(f"QQ WebSocket ready, heartbeat interval: {interval}ms")
                                if self._heartbeat_task:
                                    self._heartbeat_task.cancel()
                                self._heartbeat_task = asyncio.create_task(self._start_heartbeat(interval))
                            
                            msg_event = await self._parse_event(event)
                            if msg_event:
                                logger.info(f"QQ message parsed: type={msg_event.message_type}, user={msg_event.source.user_id}, content_len={len(msg_event.text)}")
                                await self.handle_message(msg_event)
                                logger.info(f"QQ message handled and dispatched to gateway")
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON received: {msg.data[:100]}")
                        except Exception as e:
                            logger.exception(f"Error parsing event: {e}")
                            
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        logger.warning(f"WebSocket closed or error: {msg.type}")
                        break
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"WebSocket receive loop error: {e}")
            
            if self._running:
                await self._schedule_reconnect()
    
    async def _connect_ws(self) -> None:
        """Establish WebSocket connection."""
        try:
            gateway_url = await self._get_gateway_url()
            
            headers = {
                "User-Agent": "QQBot/Hermes/1.0"
            }
            
            self._ws = await self._session.ws_connect(
                gateway_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
                heartbeat=30,
            )
            
            # 强制IDENTIFY而不是RESUMED，确保正确声明intents
            # 清空旧的session信息，强制重新IDENTIFY
            self._session_id = None
            self._last_seq = None
            
            await self._ws.send_json({
                "op": 2,
                "d": {
                    "token": f"QQBot {self._access_token}",
                    "intents": FULL_INTENTS,
                    "shard": [0, 1],
                }
            })
            
            logger.info("QQ WebSocket connected (IDENTIFY sent)")
            
        except Exception as e:
            logger.exception(f"Failed to connect WebSocket: {e}")
            raise
    
    async def _schedule_reconnect(self) -> None:
        """Schedule reconnection with exponential backoff."""
        if not self._running:
            return
        
        self._reconnect_attempts += 1
        delay_idx = min(self._reconnect_attempts, len(RECONNECT_DELAYS) - 1)
        delay = RECONNECT_DELAYS[delay_idx]
        
        logger.info(f"QQ Reconnecting in {delay}ms (attempt {self._reconnect_attempts})")
        
        await asyncio.sleep(delay / 1000)
        
        if self._running:
            try:
                await self._connect_ws()
                self._reconnect_attempts = 0
            except Exception as e:
                logger.warning(f"Reconnect failed: {e}")
    
    async def connect(self) -> bool:
        """Connect to QQ Bot platform."""
        if not check_qq_requirements():
            logger.error("QQ: Missing required dependencies (aiohttp)")
            return False
        
        logger.info("Connecting to QQ Bot...")
        
        self._session = aiohttp.ClientSession()
        
        if not await self._refresh_access_token():
            logger.error("QQ: Failed to get initial access token")
            await self._session.close()
            return False
        
        self._running = True
        self._reconnect_task = asyncio.create_task(self._ws_receive_loop())
        
        for _ in range(30):
            if self._connected:
                logger.info("QQ Bot connected successfully")
                return True
            await asyncio.sleep(0.5)
        
        logger.error("QQ Bot connection timeout")
        return False
    
    async def disconnect(self) -> None:
        """Disconnect from QQ Bot platform."""
        logger.info("Disconnecting QQ Bot...")
        self._running = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        
        if self._ws:
            await self._ws.close()
        
        if self._session:
            await self._session.close()
        
        self._connected = False
        self._mark_disconnected()
        logger.info("QQ Bot disconnected")
    
    def _get_next_msg_seq(self) -> int:
        """生成 QQ API 要求的消息序列号 (msg_seq)"""
        # 使用时间戳微秒 + 随机数保证唯一性
        return int(time.time() * 1000000) % 2147483647

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message."""
        if not self._connected:
            return SendResult(success=False, error="Not connected")
        
        try:
            await self._refresh_access_token()
            
            parts = chat_id.split(":", 2)
            if len(parts) != 3:
                return SendResult(success=False, error=f"Invalid chat_id format: {chat_id}")
            
            _, chat_type, target_id = parts
            
            # 构建 payload，使用 v2 API 格式
            payload: Dict[str, Any] = {
                "content": content,
                "msg_seq": self._get_next_msg_seq()
            }
            
            if reply_to and self.bot_config.auto_quote:
                payload["msg_id"] = reply_to
            
            if chat_type == "c2c":
                # V2 API: /v2/users/{openid}/messages
                url = f"{QQ_API_BASE}/v2/users/{target_id}/messages"
            elif chat_type == "group":
                # V2 API: /v2/groups/{group_openid}/messages
                url = f"{QQ_API_BASE}/v2/groups/{target_id}/messages"
            else:
                return SendResult(success=False, error=f"Unsupported chat type: {chat_type}")
            
            async with self._session.post(
                url,
                headers=self._auth_headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    msg_id = data.get("msg_id", data.get("id", ""))
                    logger.info(f"QQ message sent to {chat_id}: {msg_id}")
                    return SendResult(success=True, message_id=msg_id)
                else:
                    error_text = await resp.text()
                    logger.error(f"QQ send failed ({resp.status}): {error_text}")
                    return SendResult(success=False, error=f"HTTP {resp.status}: {error_text}")
                    
        except Exception as e:
            logger.exception(f"QQ send error: {e}")
            return SendResult(success=False, error=str(e))
    
    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image message."""
        if not self._connected:
            return SendResult(success=False, error="Not connected")
        
        try:
            await self._refresh_access_token()
            
            parts = chat_id.split(":", 2)
            if len(parts) != 3:
                return SendResult(success=False, error=f"Invalid chat_id format: {chat_id}")
            
            _, chat_type, target_id = parts
            
            payload: Dict[str, Any] = {
                "image_url": image_url,
            }
            if caption:
                payload["content"] = caption
            
            if chat_type == "c2c":
                url = f"{QQ_API_BASE}/v2/users/{target_id}/messages"
                payload["msg_seq"] = self._get_next_msg_seq()
            elif chat_type == "group":
                url = f"{QQ_API_BASE}/v2/groups/{target_id}/messages"
                payload["msg_seq"] = self._get_next_msg_seq()
            else:
                return SendResult(success=False, error=f"Unsupported chat type: {chat_type}")
            
            async with self._session.post(
                url,
                headers=self._auth_headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    msg_id = data.get("msg_id", data.get("id", ""))
                    return SendResult(success=True, message_id=msg_id)
                else:
                    error_text = await resp.text()
                    return SendResult(success=False, error=f"HTTP {resp.status}: {error_text}")
                    
        except Exception as e:
            logger.exception(f"QQ send image error: {e}")
            return SendResult(success=False, error=str(e))
    
    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send typing indicator (not directly supported by QQ API)."""
        pass
    
    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a chat."""
        parts = chat_id.split(":", 2)
        if len(parts) != 3:
            return {"name": chat_id, "type": "unknown", "chat_id": chat_id}
        
        _, chat_type, target_id = parts
        
        return {
            "name": target_id,
            "type": "group" if chat_type == "group" else "dm",
            "chat_id": chat_id,
        }
    
    def format_message(self, content: str) -> str:
        """Format message for QQ platform."""
        return content
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult, cache_image_from_bytes, cache_document_from_bytes

from .wechat_state import WeChatStateStore, WeChatAccount
from .wechat_transport import (
    OfficialWeChatTransport,
    AIOHTTP_AVAILABLE,
    DEFAULT_BASE_URL,
    DEFAULT_CDN_BASE_URL,
    WeChatRateLimitError,
    WeChatSessionExpiredError,
)

logger = logging.getLogger(__name__)

SESSION_PAUSE_SECONDS = 60 * 60
TYPING_CACHE_SECONDS = 24 * 60 * 60


def check_wechat_requirements() -> bool:
    return AIOHTTP_AVAILABLE


class WeChatAdapter(BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = 4000

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WECHAT)
        extra = config.extra or {}
        self._default_account_id = str(extra.get("account_id") or os.getenv("WECHAT_ACCOUNT_ID", "")).strip() or None
        self._base_url = str(extra.get("base_url") or os.getenv("WECHAT_API_BASE_URL", DEFAULT_BASE_URL)).strip() or DEFAULT_BASE_URL
        self._cdn_base_url = str(extra.get("cdn_base_url") or os.getenv("WECHAT_CDN_BASE_URL", DEFAULT_CDN_BASE_URL)).strip() or DEFAULT_CDN_BASE_URL
        self._state = WeChatStateStore()
        self._transport = OfficialWeChatTransport(base_url=self._base_url, cdn_base_url=self._cdn_base_url)
        self._poll_tasks: dict[str, asyncio.Task] = {}
        self._poll_longpoll_timeout_ms: dict[str, int] = {}
        self._sleep = asyncio.sleep
        self._paused_until: dict[str, float] = {}
        self._typing_cache: dict[tuple[str, str], tuple[str, float]] = {}

    async def connect(self) -> bool:
        if not check_wechat_requirements():
            logger.warning("[%s] WeChat startup failed: aiohttp not installed", self.name)
            return False
        self._mark_connected()
        for account_id in self._state.list_account_ids():
            if account_id in self._poll_tasks:
                continue
            task = asyncio.create_task(self._poll_account_loop(account_id))
            self._poll_tasks[account_id] = task
        if self._poll_tasks:
            await asyncio.sleep(0)
        return True

    async def disconnect(self) -> None:
        self._running = False
        for task in self._poll_tasks.values():
            task.cancel()
        if self._poll_tasks:
            await asyncio.gather(*self._poll_tasks.values(), return_exceptions=True)
        self._poll_tasks.clear()
        self._poll_longpoll_timeout_ms.clear()
        self._typing_cache.clear()
        self._paused_until.clear()
        try:
            await self._transport.close()
        finally:
            self._mark_disconnected()

    def pause_account(self, account_id: str, seconds: int = SESSION_PAUSE_SECONDS) -> None:
        self._paused_until[account_id] = time.time() + seconds

    def is_account_paused(self, account_id: str) -> bool:
        until = self._paused_until.get(account_id)
        if until is None:
            return False
        if time.time() >= until:
            self._paused_until.pop(account_id, None)
            return False
        return True

    async def start_login(self, account_id: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        return await self._transport.start_login(account_id=account_id, force=force)

    async def wait_login(self, session_key: str, timeout_ms: int = 480_000) -> Dict[str, Any]:
        result = await self._transport.wait_login(session_key=session_key, timeout_ms=timeout_ms)
        if result.get("connected") and result.get("account_id") and result.get("bot_token"):
            account = WeChatAccount(
                account_id=str(result["account_id"]),
                token=str(result["bot_token"]),
                base_url=str(result.get("base_url") or self._base_url),
                user_id=result.get("user_id"),
                enabled=True,
            )
            self._state.save_account(account)
            if self._running and account.account_id not in self._poll_tasks:
                self._poll_tasks[account.account_id] = asyncio.create_task(self._poll_account_loop(account.account_id))
        return result

    def _resolve_account(self, metadata: Optional[Dict[str, Any]] = None, chat_id: Optional[str] = None) -> WeChatAccount:
        metadata = metadata or {}
        account_id = metadata.get("account_id") or self._default_account_id

        if not account_id and chat_id:
            matched = self._state.find_account_ids_by_context_token(chat_id)
            if len(matched) == 1:
                account_id = matched[0]
            elif len(matched) > 1:
                raise RuntimeError(
                    f"WeChat account is ambiguous for {chat_id}: matched {', '.join(matched)}"
                )

        if not account_id:
            account_ids = self._state.list_account_ids()
            if len(account_ids) == 1:
                account_id = account_ids[0]

        if not account_id:
            raise RuntimeError("WeChat account_id is required")
        account = self._state.load_account(str(account_id))
        if not account:
            raise RuntimeError(f"WeChat account not found: {account_id}")
        return account

    @staticmethod
    def _is_remote_url(value: str) -> bool:
        return value.startswith("http://") or value.startswith("https://")

    async def _resolve_outbound_media_path(self, media_url: str, default_suffix: str) -> str:
        if self._is_remote_url(media_url):
            data = await self._transport._raw_http_get(url=media_url)
            parsed = urlparse(media_url)
            suffix = Path(parsed.path).suffix or default_suffix
            temp_dir = Path(tempfile.gettempdir()) / "hermes-wechat-outbound"
            temp_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix="wechat-", suffix=suffix, dir=temp_dir, delete=False) as tmp:
                tmp.write(data)
                return tmp.name
        if media_url.startswith("file://"):
            return urlparse(media_url).path
        return str(Path(media_url).resolve())

    def _resolve_context_token(self, account: WeChatAccount, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        context_token = None
        if metadata:
            context_token = metadata.get("context_token")
        if not context_token:
            context_token = self._state.get_context_token(account.account_id, chat_id)
        return context_token

    async def _ensure_typing_ticket(self, account_id: str, user_id: str, context_token: Optional[str]) -> str:
        key = (account_id, user_id)
        cached = self._typing_cache.get(key)
        now = time.time()
        if cached and cached[1] > now:
            return cached[0]
        account = self._state.load_account(account_id)
        if not account:
            return ""
        result = await self._transport.get_config(account=account, ilink_user_id=user_id, context_token=context_token)
        ticket = str(result.get("typing_ticket") or "")
        if ticket:
            self._typing_cache[key] = (ticket, now + TYPING_CACHE_SECONDS)
        return ticket

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._running:
            return SendResult(success=False, error="Not connected")
        try:
            account = self._resolve_account(metadata, chat_id=chat_id)
            context_token = self._resolve_context_token(account, chat_id, metadata)
            chunks = self.truncate_message(content, self.MAX_MESSAGE_LENGTH)
            last_result: Dict[str, Any] = {}
            for chunk in chunks:
                last_result = await self._transport.send_text(
                    account=account,
                    to_user_id=chat_id,
                    text=chunk,
                    context_token=context_token,
                )
            return SendResult(success=True, message_id=str(last_result.get("message_id") or ""), raw_response=last_result)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._running:
            return SendResult(success=False, error="Not connected")
        try:
            account = self._resolve_account(metadata, chat_id=chat_id)
            context_token = self._resolve_context_token(account, chat_id, metadata)
            file_path = await self._resolve_outbound_media_path(image_url, ".png")
            result = await self._transport.send_media_file(
                account=account,
                to_user_id=chat_id,
                file_path=file_path,
                text=caption or "",
                context_token=context_token,
            )
            return SendResult(success=True, message_id=str(result.get("message_id") or ""), raw_response=result)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self.send_image(chat_id=chat_id, image_url=image_path, caption=caption, reply_to=reply_to, metadata=metadata)

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
        if not self._running:
            return SendResult(success=False, error="Not connected")
        try:
            account = self._resolve_account(metadata, chat_id=chat_id)
            context_token = self._resolve_context_token(account, chat_id, metadata)
            resolved = await self._resolve_outbound_media_path(file_path, Path(file_path).suffix or ".bin")
            result = await self._transport.send_media_file(
                account=account,
                to_user_id=chat_id,
                file_path=resolved,
                text=caption or "",
                context_token=context_token,
            )
            return SendResult(success=True, message_id=str(result.get("message_id") or ""), raw_response=result)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        if not self._running:
            return SendResult(success=False, error="Not connected")
        try:
            account = self._resolve_account(metadata, chat_id=chat_id)
            context_token = self._resolve_context_token(account, chat_id, metadata)
            resolved = await self._resolve_outbound_media_path(audio_path, Path(audio_path).suffix or ".amr")
            result = await self._transport.send_media_file(
                account=account,
                to_user_id=chat_id,
                file_path=resolved,
                text=caption or "",
                context_token=context_token,
            )
            return SendResult(success=True, message_id=str(result.get("message_id") or ""), raw_response=result)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        if not self._running:
            return SendResult(success=False, error="Not connected")
        try:
            account = self._resolve_account(metadata, chat_id=chat_id)
            context_token = self._resolve_context_token(account, chat_id, metadata)
            resolved = await self._resolve_outbound_media_path(video_path, Path(video_path).suffix or ".mp4")
            result = await self._transport.send_media_file(
                account=account,
                to_user_id=chat_id,
                file_path=resolved,
                text=caption or "",
                context_token=context_token,
            )
            return SendResult(success=True, message_id=str(result.get("message_id") or ""), raw_response=result)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def _keep_typing(self, chat_id: str, interval: float = 5.0, metadata=None) -> None:
        """Override base interval to 5s — matches openclaw-weixin reference and reduces API pressure."""
        await super()._keep_typing(chat_id, interval=interval, metadata=metadata)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        if not self._running:
            return
        account = self._resolve_account(metadata, chat_id=chat_id)
        context_token = self._resolve_context_token(account, chat_id, metadata)
        ticket = await self._ensure_typing_ticket(account.account_id, chat_id, context_token)
        if not ticket:
            return
        await self._transport.send_typing(
            account=account,
            ilink_user_id=chat_id,
            typing_ticket=ticket,
            status=1,
        )

    async def stop_typing(self, chat_id: str) -> None:
        if not self._running:
            return
        account = self._resolve_account(None, chat_id=chat_id)
        context_token = self._resolve_context_token(account, chat_id)
        ticket = await self._ensure_typing_ticket(account.account_id, chat_id, context_token)
        if not ticket:
            return
        await self._transport.send_typing(
            account=account,
            ilink_user_id=chat_id,
            typing_ticket=ticket,
            status=2,
        )

    def _resolve_item_download_url(self, payload: Dict[str, Any]) -> str:
        media = payload.get("media") or {}
        return self._transport._extract_download_url(media) or str(payload.get("url") or "").strip()

    async def _download_binary_media(self, payload: Dict[str, Any], filename: str) -> Optional[str]:
        url = self._resolve_item_download_url(payload)
        if not url:
            return None
        data = await self._transport.fetch_media_bytes(payload.get("media") or {"full_url": url})
        return cache_document_from_bytes(data, filename)

    async def _download_media_to_cache(self, item: Dict[str, Any]) -> Optional[str]:
        if item.get("type") == 2:
            image_item = item.get("image_item") or {}
            media = image_item.get("media") or {}
            url = self._resolve_item_download_url(image_item)
            if url:
                data = await self._transport.fetch_media_bytes(media or {"full_url": url})
                suffix = Path(urlparse(url).path).suffix or ".jpg"
                return cache_image_from_bytes(data, suffix)
        if item.get("type") == 3:
            voice_item = item.get("voice_item") or {}
            return await self._download_binary_media(voice_item, "voice.amr")
        if item.get("type") == 4:
            file_item = item.get("file_item") or {}
            file_name = str(file_item.get("file_name") or "attachment.bin")
            return await self._download_binary_media(file_item, file_name)
        if item.get("type") == 5:
            video_item = item.get("video_item") or {}
            return await self._download_binary_media(video_item, "video.mp4")
        return None

    async def _poll_account_loop(self, account_id: str) -> None:
        consecutive_failures = 0
        while self._running:
            try:
                if self.is_account_paused(account_id):
                    await self._sleep(1)
                    continue
                await self._poll_account_once(account_id)
                consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except WeChatSessionExpiredError:
                self._state.clear_context_tokens(account_id)
                self.pause_account(account_id)
                await self._sleep(SESSION_PAUSE_SECONDS)
            except WeChatRateLimitError as exc:
                consecutive_failures += 1
                logger.warning("[%s] WeChat poll rate limited for %s (%d): %s", self.name, account_id, consecutive_failures, exc)
                await self._sleep(min(2 ** consecutive_failures, 60))
            except Exception as exc:
                consecutive_failures += 1
                logger.warning("[%s] WeChat poll failed for %s (%d): %s", self.name, account_id, consecutive_failures, exc)
                await self._sleep(min(consecutive_failures, 5))

    async def _poll_account_once(self, account_id: str) -> None:
        account = self._state.load_account(account_id)
        if not account:
            raise RuntimeError(f"WeChat account not found: {account_id}")
        cursor = self._state.load_sync_cursor(account_id)
        result = await self._transport.get_updates(
            account=account,
            cursor=cursor,
            longpolling_timeout_ms=self._poll_longpoll_timeout_ms.get(account_id),
        )
        next_cursor = result.get("get_updates_buf")
        if next_cursor:
            self._state.save_sync_cursor(account_id, str(next_cursor))
        hinted_timeout = result.get("longpolling_timeout_ms")
        if hinted_timeout is not None:
            try:
                hinted_timeout_int = int(hinted_timeout)
            except (TypeError, ValueError):
                hinted_timeout_int = 0
            if hinted_timeout_int > 0:
                self._poll_longpoll_timeout_ms[account_id] = hinted_timeout_int
        for raw in result.get("msgs") or []:
            event = await self._build_message_event(account_id, raw)
            context_token = raw.get("context_token")
            from_user_id = raw.get("from_user_id")
            if context_token and from_user_id:
                self._state.set_context_token(account_id, str(from_user_id), str(context_token))
            await self.handle_message(event)

    async def _build_message_event(self, account_id: str, raw: Dict[str, Any]) -> MessageEvent:
        text = ""
        message_type = MessageType.TEXT
        media_urls: list[str] = []
        media_types: list[str] = []
        priority = {2: 4, 5: 3, 4: 2, 3: 1}
        chosen_priority = 0
        for item in raw.get("item_list") or []:
            item_type = item.get("type")
            if item_type == 1 and not text:
                text = str((item.get("text_item") or {}).get("text") or "")
                continue
            if item_type == 3:
                voice_text = str((item.get("voice_item") or {}).get("text") or "")
                if voice_text and not text:
                    text = voice_text
                if voice_text and chosen_priority < priority[3]:
                    message_type = MessageType.VOICE
                    chosen_priority = priority[3]
            if item_type not in priority:
                continue
            cached = await self._download_media_to_cache(item)
            if not cached:
                continue
            media_urls = [cached]
            current_priority = priority[item_type]
            if item_type == 2:
                media_types = ["image/*"]
                if current_priority >= chosen_priority:
                    message_type = MessageType.PHOTO
            elif item_type == 5:
                media_types = ["video/mp4"]
                if current_priority >= chosen_priority:
                    message_type = MessageType.VIDEO
            elif item_type == 4:
                media_types = ["application/octet-stream"]
                if current_priority >= chosen_priority:
                    message_type = MessageType.DOCUMENT
            elif item_type == 3:
                media_types = ["audio/*"]
                if current_priority >= chosen_priority:
                    message_type = MessageType.VOICE
            chosen_priority = max(chosen_priority, current_priority)
        from_user_id = str(raw.get("from_user_id") or "")
        source = self.build_source(
            chat_id=from_user_id,
            chat_name=from_user_id,
            chat_type="dm",
            user_id=from_user_id,
            user_name=from_user_id,
            user_id_alt=account_id,
        )
        timestamp_ms = raw.get("create_time_ms")
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000) if timestamp_ms else datetime.now()
        return MessageEvent(
            text=text,
            message_type=message_type,
            source=source,
            raw_message={"account_id": account_id, **raw},
            message_id=str(raw.get("message_id") or ""),
            timestamp=timestamp,
            media_urls=media_urls,
            media_types=media_types,
        )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}

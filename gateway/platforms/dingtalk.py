"""
DingTalk platform adapter using Stream Mode.

Uses dingtalk-stream Python SDK for:
- Receiving messages via Stream mode (no webhook required)
- Sending responses with markdown/AI cards
- Handling media (images, files, audio, video)
- AI Card streaming for real-time responses

Requirements:
    pip install dingtalk-stream

Configuration:
    DINGTALK_CLIENT_ID: AppKey from https://open-dev.dingtalk.com
    DINGTALK_CLIENT_SECRET: AppSecret from https://open-dev.dingtalk.com
"""

import asyncio
import logging
import os
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import dingtalk-stream SDK
try:
    import dingtalk_stream
    from dingtalk_stream import (
        DingTalkStreamClient,
        Credential,
        ChatbotMessage,
        ChatbotHandler,
        AckMessage,
        AICardReplier,
        AICardStatus,
        AIMarkdownCardInstance,
        MarkdownCardInstance,
    )
    DINGTALK_AVAILABLE = True
except ImportError:
    DINGTALK_AVAILABLE = False
    DingTalkStreamClient = Any
    Credential = Any
    ChatbotMessage = Any
    ChatbotHandler = Any
    AckMessage = Any
    AICardReplier = Any
    AICardStatus = Any
    AIMarkdownCardInstance = Any
    MarkdownCardInstance = Any

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_document_from_bytes,
)


def check_dingtalk_requirements() -> bool:
    """Check if DingTalk dependencies are available."""
    return DINGTALK_AVAILABLE


class DingTalkAdapter(BasePlatformAdapter):
    """
    DingTalk bot adapter using Stream mode.
    
    Features:
    - Stream mode connection (no public webhook URL needed)
    - AI Card streaming for real-time responses
    - Markdown and text message support
    - Media handling (images, files, audio, video)
    - Single chat and group chat support
    """
    
    MAX_MESSAGE_LENGTH = 20000
    MAX_MARKDOWN_LENGTH = 50000
    
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.DINGTALK)
        self._client = None
        self._handler = None
        self._client_id = None
        self._client_secret = None
        self._access_token = None
        self._access_token_expiry = 0
        self._active_cards = {}
        self._stream_task = None
        self._processed_messages = set()  # Message ID deduplication
        self._processed_messages_lock = None  # Will be set in connect()
        
    @property
    def name(self) -> str:
        return "DingTalk"
    
    def is_message_processed(self, message_id: str) -> bool:
        """Check if a message has already been processed (deduplication)."""
        if self._processed_messages_lock is None:
            return message_id in self._processed_messages
        import asyncio
        if asyncio.iscoroutinecontext():
            # We're in async context but this is a sync method
            return message_id in self._processed_messages
        return message_id in self._processed_messages
    
    def mark_message_processed(self, message_id: str) -> None:
        """Mark a message as processed to prevent duplicate handling."""
        self._processed_messages.add(message_id)
        # Keep only last 1000 message IDs to prevent memory growth
        if len(self._processed_messages) > 1000:
            # Remove oldest half
            to_remove = list(self._processed_messages)[:500]
            for mid in to_remove:
                self._processed_messages.discard(mid)
    
    def _get_client_id(self) -> Optional[str]:
        if self._client_id:
            return self._client_id
        self._client_id = self.config.extra.get("client_id") or os.getenv("DINGTALK_CLIENT_ID")
        return self._client_id
    
    def _get_client_secret(self) -> Optional[str]:
        if self._client_secret:
            return self._client_secret
        self._client_secret = self.config.extra.get("client_secret") or os.getenv("DINGTALK_CLIENT_SECRET")
        return self._client_secret
    
    async def _get_access_token(self) -> Optional[str]:
        import time
        now = int(time.time())
        
        if self._access_token and self._access_token_expiry > now + 60:
            return self._access_token
        
        client_id = self._get_client_id()
        client_secret = self._get_client_secret()
        if not client_id or not client_secret:
            return None
        
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://api.dingtalk.com/v1.0/oauth2/accessToken",
                    json={"appKey": client_id, "appSecret": client_secret}
                )
                data = resp.json()
                self._access_token = data.get("accessToken")
                self._access_token_expiry = now + data.get("expireIn", 7200)
                return self._access_token
        except Exception as e:
            logger.error("[%s] Failed to get access token: %s", self.name, e)
            return None
    
    async def connect(self) -> bool:
        if not DINGTALK_AVAILABLE:
            logger.error(
                "[%s] dingtalk-stream not installed. Run: pip install dingtalk-stream",
                self.name,
            )
            return False
        
        client_id = self._get_client_id()
        client_secret = self._get_client_secret()
        
        if not client_id or not client_secret:
            logger.error("[%s] No client_id/client_secret configured", self.name)
            return False
        
        try:
            credential = Credential(client_id, client_secret)
            self._client = DingTalkStreamClient(credential)
            self._handler = DingTalkMessageHandler(self)
            self._client.register_callback_handler(
                ChatbotMessage.TOPIC,
                self._handler
            )
            self._stream_task = asyncio.create_task(self._run_stream_client())
            self._mark_connected()
            logger.info("[%s] Connected to DingTalk via Stream mode", self.name)
            return True
        except Exception as e:
            logger.error("[%s] Failed to connect to DingTalk: %s", self.name, e, exc_info=True)
            return False
    
    async def _run_stream_client(self):
        while self._running:
            try:
                await self._client.start()
            except asyncio.CancelledError:
                logger.info("[%s] Stream client cancelled", self.name)
                break
            except Exception as e:
                logger.error("[%s] Stream client error: %s", self.name, e, exc_info=True)
                if self._running:
                    logger.info("[%s] Reconnecting in 5 seconds...", self.name)
                    await asyncio.sleep(5)
    
    async def disconnect(self) -> None:
        self._running = False
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None
        self._client = None
        self._handler = None
        self._active_cards.clear()
        self._mark_disconnected()
        logger.info("[%s] Disconnected from DingTalk", self.name)
    
    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        if not self._handler:
            return SendResult(success=False, error="Not connected")
        
        try:
            # Use raw_message from metadata to get session_webhook for direct reply
            raw_message = metadata.get("raw_message") if metadata else None
            logger.info("[%s] send() called: raw_message=%s, has_session_webhook=%s", 
                        self.name, type(raw_message).__name__ if raw_message else None,
                        hasattr(raw_message, 'session_webhook') if raw_message else False)
            
            if raw_message and hasattr(raw_message, 'session_webhook') and raw_message.session_webhook:
                logger.info("[%s] Using reply_markdown with session_webhook: %s", self.name, raw_message.session_webhook[:50] + '...')
                result = self._handler.reply_markdown(
                    title="Hermes",
                    text=content,
                    incoming_message=raw_message
                )
                logger.info("[%s] reply_markdown result: %s", self.name, result)
                return SendResult(success=True)
            
            # Fallback: construct incoming message for reply
            logger.info("[%s] No session_webhook, using fallback")
            chat_type = metadata.get("chat_type", "direct") if metadata else "direct"
            conversation_id = metadata.get("conversation_id") if metadata else None
            
            if chat_type == "group" and conversation_id:
                incoming = dingtalk_stream.reply_specified_group_chat(conversation_id)
            else:
                incoming = dingtalk_stream.reply_specified_single_chat(
                    chat_id, 
                    metadata.get("sender_name", "") if metadata else ""
                )
            
            # Try reply_markdown first (no card permission needed)
            result = self._handler.reply_markdown(
                title="Hermes",
                text=content,
                incoming_message=incoming
            )
            logger.info("[%s] Fallback reply_markdown result: %s", self.name, result)
            return SendResult(success=True)
        except Exception as e:
            logger.error("[%s] Failed to send message: %s", self.name, e, exc_info=True)
            return SendResult(success=False, error=str(e))
    
    async def send_typing(self, chat_id: str, metadata=None) -> None:
        pass
    
    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        if not self._handler:
            return SendResult(success=False, error="Not connected")
        
        try:
            token = await self._get_access_token()
            if not token:
                return SendResult(success=False, error="Failed to get access token")
            
            media_id = await self._upload_media(image_path, "image", token)
            if not media_id:
                return SendResult(success=False, error="Failed to upload image")
            
            metadata = kwargs.get("metadata", {})
            chat_type = metadata.get("chat_type", "direct")
            conversation_id = metadata.get("conversation_id")
            
            if chat_type == "group" and conversation_id:
                incoming = dingtalk_stream.reply_specified_group_chat(conversation_id)
            else:
                incoming = dingtalk_stream.reply_specified_single_chat(chat_id)
            
            replier = dingtalk_stream.CardReplier(self._client, incoming)
            await replier.reply_image(media_id)
            return SendResult(success=True)
        except Exception as e:
            logger.error("[%s] Failed to send image: %s", self.name, e, exc_info=True)
            return SendResult(success=False, error=str(e))
    
    async def _upload_media(self, file_path: str, media_type: str, access_token: str) -> Optional[str]:
        try:
            import httpx
            if not os.path.exists(file_path):
                return None
            
            file_size = os.path.getsize(file_path)
            if file_size > 20 * 1024 * 1024:
                return None
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(file_path, "rb") as f:
                    files = {"media": (os.path.basename(file_path), f)}
                    resp = await client.post(
                        f"https://oapi.dingtalk.com/media/upload?access_token={access_token}&type={media_type}",
                        files=files
                    )
                    data = resp.json()
                    if data.get("errcode", 0) == 0:
                        return data.get("media_id")
            return None
        except Exception as e:
            logger.error("[%s] Upload error: %s", self.name, e)
            return None
    
    async def start_streaming_card(self, incoming_message, title: str = ""):
        if not self._handler:
            return None
        try:
            card = self._handler.ai_markdown_card_start(
                incoming_message=incoming_message,
                title=title,
            )
            self._active_cards[incoming_message.message_id] = card
            return card
        except Exception as e:
            logger.error("[%s] Failed to start AI Card: %s", self.name, e)
            return None
    
    async def stream_to_card(self, message_id: str, content: str, append: bool = True) -> bool:
        card = self._active_cards.get(message_id)
        if not card:
            return False
        try:
            card.ai_streaming(markdown=content, append=append)
            return True
        except Exception as e:
            logger.error("[%s] Failed to stream to card: %s", self.name, e)
            return False
    
    async def finish_streaming_card(self, message_id: str, final_content: str) -> bool:
        card = self._active_cards.pop(message_id, None)
        if not card:
            return False
        try:
            card.ai_finish(markdown=final_content)
            return True
        except Exception as e:
            logger.error("[%s] Failed to finish card: %s", self.name, e)
            return False
    
    def format_message(self, content: str) -> str:
        return content
    
    def truncate_message(self, content: str, max_length: int = None) -> List[str]:
        max_length = max_length or self.MAX_MESSAGE_LENGTH
        if len(content) <= max_length:
            return [content]
        
        chunks = []
        paragraphs = content.split("\n\n")
        current = ""
        
        for para in paragraphs:
            if len(current) + len(para) + 2 <= max_length:
                current = current + "\n\n" + para if current else para
            else:
                if current:
                    chunks.append(current)
                if len(para) > max_length:
                    for i in range(0, len(para), max_length):
                        chunks.append(para[i:i + max_length])
                else:
                    current = para
        
        if current:
            chunks.append(current)
        return chunks
    
    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a chat."""
        return {
            "name": chat_id,
            "type": "dm",  # Default to DM, actual type determined by conversation_type
            "chat_id": chat_id,
        }


class DingTalkMessageHandler(ChatbotHandler):
    """Handler for incoming DingTalk messages."""
    
    def __init__(self, adapter: DingTalkAdapter):
        super().__init__()
        self.adapter = adapter
        self.logger = logger
    
    async def process(self, callback):
        try:
            incoming_message = ChatbotMessage.from_dict(callback.data)
            event = self._create_message_event(incoming_message)
            if not event:
                return AckMessage.STATUS_OK, "OK"
            
            if self.adapter.is_message_processed(event.message_id):
                return AckMessage.STATUS_OK, "OK"
            
            self.adapter.mark_message_processed(event.message_id)
            await self.adapter.handle_message(event)
            return AckMessage.STATUS_OK, "OK"
        except Exception as e:
            self.logger.error("[%s] Error processing message: %s", self.adapter.name, e)
            return AckMessage.STATUS_OK, "OK"
    
    def _create_message_event(self, incoming) -> Optional[MessageEvent]:
        from gateway.session import SessionSource
        
        text = ""
        message_type = MessageType.TEXT
        media_urls = []
        media_types = []
        
        if incoming.message_type == "text":
            text = incoming.text.content.strip() if incoming.text else ""
            message_type = MessageType.TEXT
        elif incoming.message_type == "picture":
            message_type = MessageType.PHOTO
            if incoming.image_content and incoming.image_content.download_code:
                media_urls.append(incoming.image_content.download_code)
                media_types.append("image")
        elif incoming.message_type == "richText":
            message_type = MessageType.TEXT
            if incoming.rich_text_content and incoming.rich_text_content.rich_text_list:
                text_parts = []
                for item in incoming.rich_text_content.rich_text_list:
                    if "text" in item:
                        text_parts.append(item["text"])
                    if "downloadCode" in item:
                        media_urls.append(item["downloadCode"])
                        media_types.append("image")
                text = " ".join(text_parts)
        
        if not text and not media_urls:
            return None
        
        is_direct = incoming.conversation_type == "1"
        chat_type = "direct" if is_direct else "group"
        sender_id = incoming.sender_staff_id or incoming.sender_id
        
        source = SessionSource(
            platform=Platform.DINGTALK,
            chat_id=incoming.conversation_id if not is_direct else sender_id,
            chat_type=chat_type,
            user_id=sender_id,
            user_name=incoming.sender_nick or "Unknown",
            thread_id=None,
        )
        
        return MessageEvent(
            text=text,
            message_type=message_type,
            source=source,
            raw_message=incoming,
            message_id=incoming.message_id,
            media_urls=media_urls,
            media_types=media_types,
            timestamp=datetime.now(),
        )
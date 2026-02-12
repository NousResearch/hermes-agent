"""
Telegram platform adapter.

Uses python-telegram-bot library for:
- Receiving messages from users/groups
- Sending responses back
- Handling media and commands
"""

import asyncio
from typing import Dict, List, Optional, Any

try:
    from telegram import Update, Bot, Message
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler as TelegramMessageHandler,
        ContextTypes,
        filters,
    )
    from telegram.constants import ParseMode, ChatType
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = Any
    Bot = Any
    Message = Any
    Application = Any
    ContextTypes = Any

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)


def check_telegram_requirements() -> bool:
    """Check if Telegram dependencies are available."""
    return TELEGRAM_AVAILABLE


class TelegramAdapter(BasePlatformAdapter):
    """
    Telegram bot adapter.
    
    Handles:
    - Receiving messages from users and groups
    - Sending responses with Telegram markdown
    - Forum topics (thread_id support)
    - Media messages
    """
    
    # Telegram message limits
    MAX_MESSAGE_LENGTH = 4096
    
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.TELEGRAM)
        self._app: Optional[Application] = None
        self._bot: Optional[Bot] = None
    
    async def connect(self) -> bool:
        """Connect to Telegram and start polling for updates."""
        if not TELEGRAM_AVAILABLE:
            print(f"[{self.name}] python-telegram-bot not installed. Run: pip install python-telegram-bot")
            return False
        
        if not self.config.token:
            print(f"[{self.name}] No bot token configured")
            return False
        
        try:
            # Build the application
            self._app = Application.builder().token(self.config.token).build()
            self._bot = self._app.bot
            
            # Register handlers
            self._app.add_handler(TelegramMessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._handle_text_message
            ))
            self._app.add_handler(TelegramMessageHandler(
                filters.COMMAND,
                self._handle_command
            ))
            self._app.add_handler(TelegramMessageHandler(
                filters.PHOTO | filters.VIDEO | filters.AUDIO | filters.VOICE | filters.Document.ALL,
                self._handle_media_message
            ))
            
            # Start polling in background
            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
            
            self._running = True
            print(f"[{self.name}] Connected and polling for updates")
            return True
            
        except Exception as e:
            print(f"[{self.name}] Failed to connect: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Stop polling and disconnect."""
        if self._app:
            try:
                await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception as e:
                print(f"[{self.name}] Error during disconnect: {e}")
        
        self._running = False
        self._app = None
        self._bot = None
        print(f"[{self.name}] Disconnected")
    
    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        """Send a message to a Telegram chat."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        
        try:
            # Format and split message if needed
            formatted = self.format_message(content)
            chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)
            
            message_ids = []
            thread_id = metadata.get("thread_id") if metadata else None
            
            for i, chunk in enumerate(chunks):
                # Try Markdown first, fall back to plain text if it fails
                try:
                    msg = await self._bot.send_message(
                        chat_id=int(chat_id),
                        text=chunk,
                        parse_mode=ParseMode.MARKDOWN,
                        reply_to_message_id=int(reply_to) if reply_to and i == 0 else None,
                        message_thread_id=int(thread_id) if thread_id else None,
                    )
                except Exception as md_error:
                    # Markdown parsing failed, try plain text
                    if "parse" in str(md_error).lower() or "markdown" in str(md_error).lower():
                        msg = await self._bot.send_message(
                            chat_id=int(chat_id),
                            text=chunk,
                            parse_mode=None,  # Plain text
                            reply_to_message_id=int(reply_to) if reply_to and i == 0 else None,
                            message_thread_id=int(thread_id) if thread_id else None,
                        )
                    else:
                        raise  # Re-raise if not a parse error
                message_ids.append(str(msg.message_id))
            
            return SendResult(
                success=True,
                message_id=message_ids[0] if message_ids else None,
                raw_response={"message_ids": message_ids}
            )
            
        except Exception as e:
            return SendResult(success=False, error=str(e))
    
    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """Send an image natively as a Telegram photo."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        
        try:
            # Telegram can send photos directly from URLs
            msg = await self._bot.send_photo(
                chat_id=int(chat_id),
                photo=image_url,
                caption=caption[:1024] if caption else None,  # Telegram caption limit
                reply_to_message_id=int(reply_to) if reply_to else None,
            )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            print(f"[{self.name}] Failed to send photo, falling back to URL: {e}")
            # Fallback: send as text link
            return await super().send_image(chat_id, image_url, caption, reply_to)
    
    async def send_typing(self, chat_id: str) -> None:
        """Send typing indicator."""
        if self._bot:
            try:
                await self._bot.send_chat_action(
                    chat_id=int(chat_id),
                    action="typing"
                )
            except Exception:
                pass  # Ignore typing indicator failures
    
    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a Telegram chat."""
        if not self._bot:
            return {"name": "Unknown", "type": "dm"}
        
        try:
            chat = await self._bot.get_chat(int(chat_id))
            
            chat_type = "dm"
            if chat.type == ChatType.GROUP:
                chat_type = "group"
            elif chat.type == ChatType.SUPERGROUP:
                chat_type = "group"
                if chat.is_forum:
                    chat_type = "forum"
            elif chat.type == ChatType.CHANNEL:
                chat_type = "channel"
            
            return {
                "name": chat.title or chat.full_name or str(chat_id),
                "type": chat_type,
                "username": chat.username,
                "is_forum": getattr(chat, "is_forum", False),
            }
        except Exception as e:
            return {"name": str(chat_id), "type": "dm", "error": str(e)}
    
    def format_message(self, content: str) -> str:
        """
        Format message for Telegram.
        
        Telegram uses a subset of markdown. We'll use the simpler
        Markdown mode (not MarkdownV2) for compatibility.
        """
        # Basic escaping for Telegram Markdown
        # In Markdown mode (not V2), only certain characters need escaping
        return content
    
    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text messages."""
        if not update.message or not update.message.text:
            return
        
        event = self._build_message_event(update.message, MessageType.TEXT)
        await self.handle_message(event)
    
    async def _handle_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming command messages."""
        if not update.message or not update.message.text:
            return
        
        event = self._build_message_event(update.message, MessageType.COMMAND)
        await self.handle_message(event)
    
    async def _handle_media_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming media messages."""
        if not update.message:
            return
        
        msg = update.message
        
        # Determine media type
        if msg.photo:
            msg_type = MessageType.PHOTO
        elif msg.video:
            msg_type = MessageType.VIDEO
        elif msg.audio:
            msg_type = MessageType.AUDIO
        elif msg.voice:
            msg_type = MessageType.VOICE
        else:
            msg_type = MessageType.DOCUMENT
        
        event = self._build_message_event(msg, msg_type)
        
        # Add caption as text
        if msg.caption:
            event.text = msg.caption
        
        await self.handle_message(event)
    
    def _build_message_event(self, message: Message, msg_type: MessageType) -> MessageEvent:
        """Build a MessageEvent from a Telegram message."""
        chat = message.chat
        user = message.from_user
        
        # Determine chat type
        chat_type = "dm"
        if chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
            chat_type = "group"
        elif chat.type == ChatType.CHANNEL:
            chat_type = "channel"
        
        # Build source
        source = self.build_source(
            chat_id=str(chat.id),
            chat_name=chat.title or (chat.full_name if hasattr(chat, "full_name") else None),
            chat_type=chat_type,
            user_id=str(user.id) if user else None,
            user_name=user.full_name if user else None,
            thread_id=str(message.message_thread_id) if message.message_thread_id else None,
        )
        
        return MessageEvent(
            text=message.text or "",
            message_type=msg_type,
            source=source,
            raw_message=message,
            message_id=str(message.message_id),
            timestamp=message.date,
        )

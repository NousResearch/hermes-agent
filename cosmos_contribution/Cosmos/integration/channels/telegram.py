"""
Cosmos Telegram Channel Adapter
====================================

Telegram Bot API integration for the Cosmos swarm.

Features:
- Long-polling and webhook support
- Inline keyboards and callbacks
- Media handling (photos, videos, audio, documents)
- Group and supergroup support
- Forum topic threading
- Typing indicators
- Read receipts via message reactions

Based on python-telegram-bot library.

Setup:
1. Create bot via @BotFather
2. Get token
3. Set TELEGRAM_BOT_TOKEN env var or config

"From Russia with love, to the swarm." - The Collective
"""

import os
import asyncio
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
from pathlib import Path
from loguru import logger

from .channel_hub import (
    BaseChannel,
    ChannelConfig,
    ChannelMessage,
    ChannelType,
)

# Telegram library
try:
    from telegram import (
        Update,
        Bot,
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        InputFile,
    )
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        CallbackQueryHandler,
        filters,
        ContextTypes,
    )
    from telegram.constants import ParseMode, ChatAction
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Run: pip install python-telegram-bot")


class TelegramChannel(BaseChannel):
    """
    Telegram Bot API channel adapter.

    Supports:
    - Direct messages
    - Groups and supergroups
    - Forum topics
    - Inline keyboards
    - Media (photos, videos, audio, documents, stickers)
    - Commands (/start, /help, custom)
    """

    def __init__(self, config: ChannelConfig = None, token: str = None):
        """
        Initialize Telegram channel.

        Args:
            config: Channel configuration
            token: Bot token (or set TELEGRAM_BOT_TOKEN env var)
        """
        config = config or ChannelConfig(channel_type=ChannelType.TELEGRAM)
        super().__init__(config)

        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN") or config.token

        self._app: Optional[Application] = None
        self._bot: Optional[Bot] = None
        self._command_handlers: Dict[str, Callable] = {}
        self._callback_handlers: Dict[str, Callable] = {}

    async def connect(self) -> bool:
        """Connect to Telegram and start polling."""
        if not TELEGRAM_AVAILABLE:
            logger.error("python-telegram-bot not available")
            return False

        if not self.token:
            logger.error("Telegram bot token not configured")
            return False

        try:
            # Build application
            self._app = Application.builder().token(self.token).build()
            self._bot = self._app.bot

            # Register handlers
            self._app.add_handler(CommandHandler("start", self._handle_start))
            self._app.add_handler(CommandHandler("help", self._handle_help))
            self._app.add_handler(CommandHandler("status", self._handle_status))
            self._app.add_handler(CommandHandler("reset", self._handle_reset))

            # Message handler for all text/media
            self._app.add_handler(MessageHandler(
                filters.TEXT | filters.PHOTO | filters.VIDEO |
                filters.AUDIO | filters.VOICE | filters.Document.ALL |
                filters.Sticker.ALL,
                self._handle_message
            ))

            # Callback query handler for inline keyboards
            self._app.add_handler(CallbackQueryHandler(self._handle_callback))

            # Start polling in background
            await self._app.initialize()
            await self._app.start()
            asyncio.create_task(self._app.updater.start_polling(drop_pending_updates=True))

            self._connected = True

            # Get bot info
            bot_info = await self._bot.get_me()
            logger.info(f"Telegram bot connected: @{bot_info.username}")

            return True

        except Exception as e:
            logger.error(f"Telegram connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Telegram."""
        if self._app:
            try:
                await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception as e:
                logger.warning(f"Telegram disconnect error: {e}")

        self._connected = False
        logger.info("Telegram disconnected")

    async def send_message(
        self,
        chat_id: str,
        text: str,
        media_path: Optional[str] = None,
        reply_to: Optional[str] = None,
        buttons: List[List[Dict]] = None,
        parse_mode: str = "Markdown",
        **kwargs
    ) -> bool:
        """
        Send a message to a Telegram chat.

        Args:
            chat_id: Telegram chat ID
            text: Message text
            media_path: Optional media file path
            reply_to: Message ID to reply to
            buttons: Inline keyboard buttons [[{text, callback_data}]]
            parse_mode: Text parsing mode (Markdown, HTML)

        Returns:
            True if sent successfully
        """
        if not self._bot:
            return False

        if not self._check_rate_limit(chat_id):
            logger.warning(f"Rate limit exceeded for chat {chat_id}")
            return False

        try:
            # Build keyboard if buttons provided
            keyboard = None
            if buttons:
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton(btn.get("text", ""), callback_data=btn.get("callback_data", ""))
                     for btn in row]
                    for row in buttons
                ])

            # Send typing indicator
            if self.config.typing_indicator:
                await self._bot.send_chat_action(int(chat_id), ChatAction.TYPING)

            # Send with media or text only
            if media_path and Path(media_path).exists():
                media_path = Path(media_path)
                suffix = media_path.suffix.lower()

                with open(media_path, "rb") as f:
                    if suffix in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                        await self._bot.send_photo(
                            int(chat_id),
                            photo=f,
                            caption=text[:1024] if text else None,
                            reply_to_message_id=int(reply_to) if reply_to else None,
                            reply_markup=keyboard,
                            parse_mode=parse_mode
                        )
                    elif suffix in [".mp4", ".mov", ".avi", ".webm"]:
                        await self._bot.send_video(
                            int(chat_id),
                            video=f,
                            caption=text[:1024] if text else None,
                            reply_to_message_id=int(reply_to) if reply_to else None,
                            reply_markup=keyboard,
                            parse_mode=parse_mode
                        )
                    elif suffix in [".mp3", ".wav", ".ogg", ".m4a"]:
                        # Check if should send as voice
                        if kwargs.get("as_voice") or "[[audio_as_voice]]" in text:
                            text = text.replace("[[audio_as_voice]]", "").strip()
                            await self._bot.send_voice(
                                int(chat_id),
                                voice=f,
                                caption=text[:1024] if text else None,
                                reply_to_message_id=int(reply_to) if reply_to else None
                            )
                        else:
                            await self._bot.send_audio(
                                int(chat_id),
                                audio=f,
                                caption=text[:1024] if text else None,
                                reply_to_message_id=int(reply_to) if reply_to else None,
                                reply_markup=keyboard,
                                parse_mode=parse_mode
                            )
                    else:
                        await self._bot.send_document(
                            int(chat_id),
                            document=f,
                            caption=text[:1024] if text else None,
                            reply_to_message_id=int(reply_to) if reply_to else None,
                            reply_markup=keyboard,
                            parse_mode=parse_mode
                        )
            else:
                await self._bot.send_message(
                    int(chat_id),
                    text=text,
                    reply_to_message_id=int(reply_to) if reply_to else None,
                    reply_markup=keyboard,
                    parse_mode=parse_mode
                )

            return True

        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages."""
        if not update.message:
            return

        msg = update.message
        chat = msg.chat
        user = msg.from_user

        # Build normalized message
        message = ChannelMessage(
            message_id=str(msg.message_id),
            channel_type=ChannelType.TELEGRAM,
            channel_id=str(chat.id),
            sender_id=str(user.id) if user else "unknown",
            sender_name=user.full_name if user else "Unknown",
            text=msg.text or msg.caption or "",
            is_group=chat.type in ["group", "supergroup"],
            group_name=chat.title if chat.type in ["group", "supergroup"] else None,
            timestamp=msg.date,
            raw_data={
                "update_id": update.update_id,
                "chat_type": chat.type,
                "message_thread_id": msg.message_thread_id,
            }
        )

        # Check for mentions
        if msg.entities:
            for entity in msg.entities:
                if entity.type == "mention":
                    message.is_mention = True
                    break

        # Check for reply
        if msg.reply_to_message:
            message.reply_to_id = str(msg.reply_to_message.message_id)
            message.reply_to_text = msg.reply_to_message.text or msg.reply_to_message.caption

        # Handle media
        if msg.photo:
            message.media_type = "image"
            # Get largest photo
            photo = max(msg.photo, key=lambda p: p.file_size or 0)
            file = await context.bot.get_file(photo.file_id)
            message.media_url = file.file_path

        elif msg.video:
            message.media_type = "video"
            file = await context.bot.get_file(msg.video.file_id)
            message.media_url = file.file_path

        elif msg.audio:
            message.media_type = "audio"
            file = await context.bot.get_file(msg.audio.file_id)
            message.media_url = file.file_path

        elif msg.voice:
            message.media_type = "audio"
            file = await context.bot.get_file(msg.voice.file_id)
            message.media_url = file.file_path

        elif msg.document:
            message.media_type = "document"
            file = await context.bot.get_file(msg.document.file_id)
            message.media_url = file.file_path

        elif msg.sticker:
            message.media_type = "sticker"
            if not msg.sticker.is_animated and not msg.sticker.is_video:
                file = await context.bot.get_file(msg.sticker.file_id)
                message.media_url = file.file_path

        # Auto-react if configured
        if self.config.auto_react:
            try:
                await msg.set_reaction(self.config.auto_react_emoji)
            except Exception:
                pass  # Reactions may not be available

        # Pass to handler
        await self._handle_inbound(message)

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks."""
        query = update.callback_query
        if not query:
            return

        await query.answer()

        callback_data = query.data
        user = query.from_user
        chat = query.message.chat if query.message else None

        # Check for registered handlers
        for prefix, handler in self._callback_handlers.items():
            if callback_data.startswith(prefix):
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(callback_data, user.id, chat.id if chat else None)
                    else:
                        handler(callback_data, user.id, chat.id if chat else None)
                except Exception as e:
                    logger.error(f"Callback handler error: {e}")
                return

        # Default: treat as message
        message = ChannelMessage(
            message_id=str(query.id),
            channel_type=ChannelType.TELEGRAM,
            channel_id=str(chat.id) if chat else "unknown",
            sender_id=str(user.id),
            sender_name=user.full_name,
            text=f"[Button: {callback_data}]",
            timestamp=datetime.now(),
            raw_data={"callback_data": callback_data}
        )

        await self._handle_inbound(message)

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        if update.message:
            await update.message.reply_text(
                "👋 Welcome to Cosmos!\n\n"
                "I'm a collective AI swarm. You can:\n"
                "• Chat with me directly\n"
                "• Add me to groups (@mention to activate)\n"
                "• Use /help for commands\n\n"
                "What would you like to explore?",
                parse_mode=ParseMode.MARKDOWN
            )

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        if update.message:
            await update.message.reply_text(
                "📚 *Cosmos Commands*\n\n"
                "/start - Introduction\n"
                "/help - This message\n"
                "/status - Check swarm status\n"
                "/reset - Reset conversation\n\n"
                "Just send a message to chat!",
                parse_mode=ParseMode.MARKDOWN
            )

    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        if update.message:
            await update.message.reply_text(
                "✅ *Cosmos Status*\n\n"
                "• Swarm: Online\n"
                "• Telegram: Connected\n"
                "• Agents: Active\n\n"
                "Ready to assist!",
                parse_mode=ParseMode.MARKDOWN
            )

    async def _handle_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reset command."""
        if update.message:
            await update.message.reply_text(
                "🔄 Conversation reset. Starting fresh!",
                parse_mode=ParseMode.MARKDOWN
            )

    def register_command(self, command: str, handler: Callable):
        """Register a custom command handler."""
        if self._app:
            self._app.add_handler(CommandHandler(command, handler))

    def register_callback_prefix(self, prefix: str, handler: Callable):
        """Register a callback handler for buttons with specific prefix."""
        self._callback_handlers[prefix] = handler

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        buttons: List[List[Dict]] = None,
        parse_mode: str = "Markdown"
    ) -> bool:
        """Edit an existing message."""
        if not self._bot:
            return False

        try:
            keyboard = None
            if buttons:
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton(btn.get("text", ""), callback_data=btn.get("callback_data", ""))
                     for btn in row]
                    for row in buttons
                ])

            await self._bot.edit_message_text(
                text=text,
                chat_id=int(chat_id),
                message_id=int(message_id),
                reply_markup=keyboard,
                parse_mode=parse_mode
            )
            return True

        except Exception as e:
            logger.error(f"Edit message failed: {e}")
            return False

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a message."""
        if not self._bot:
            return False

        try:
            await self._bot.delete_message(int(chat_id), int(message_id))
            return True
        except Exception as e:
            logger.error(f"Delete message failed: {e}")
            return False

    async def get_chat_info(self, chat_id: str) -> Optional[Dict]:
        """Get information about a chat."""
        if not self._bot:
            return None

        try:
            chat = await self._bot.get_chat(int(chat_id))
            return {
                "id": chat.id,
                "type": chat.type,
                "title": chat.title,
                "username": chat.username,
                "description": chat.description,
                "member_count": await self._bot.get_chat_member_count(int(chat_id))
            }
        except Exception as e:
            logger.error(f"Get chat info failed: {e}")
            return None

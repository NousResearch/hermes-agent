"""
cosmos Telegram Adapter
---------------------------
Connects cosmos to Telegram via python-telegram-bot.

Usage:
    export TELEGRAM_BOT_TOKEN=your_bot_token

    from Cosmos.interfaces.messaging.telegram_adapter import TelegramAdapter

    adapter = TelegramAdapter()
    adapter.set_callback(my_message_handler)
    await adapter.connect()
"""

import os
import asyncio
from datetime import datetime
from typing import Optional, Callable, Awaitable
from loguru import logger

from Cosmos.interfaces.messaging.base import (
    MessagingProvider,
    IncomingMessage,
    OutgoingMessage,
    MessageSource
)

# Try to import telegram
try:
    from telegram import Update, Bot
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Run: pip install python-telegram-bot")


class TelegramAdapter(MessagingProvider):
    """
    Telegram messaging adapter for cosmos.

    Provides:
    - Two-way messaging with Telegram users
    - Command handling (/start, /help, /ask)
    - Photo/document support (future)
    - Group chat support
    """

    def __init__(self, token: Optional[str] = None):
        super().__init__("telegram")
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.app: Optional[Application] = None
        self.bot: Optional[Bot] = None
        self._running = False

        # cosmos persona
        self.persona_prefix = "*adjusts spectacles* "

    async def connect(self):
        """Start the Telegram bot."""
        if not TELEGRAM_AVAILABLE:
            logger.error("python-telegram-bot not installed")
            return

        if not self.token:
            logger.error("TELEGRAM_BOT_TOKEN not set")
            return

        try:
            # Build application
            self.app = Application.builder().token(self.token).build()
            self.bot = self.app.bot

            # Add handlers
            self.app.add_handler(CommandHandler("start", self._handle_start))
            self.app.add_handler(CommandHandler("help", self._handle_help))
            self.app.add_handler(CommandHandler("ask", self._handle_ask))
            self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

            # Start polling
            self._running = True
            logger.info("Telegram bot connected and listening!")

            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()

        except Exception as e:
            logger.error(f"Failed to connect Telegram bot: {e}")
            self._running = False

    async def disconnect(self):
        """Stop the Telegram bot."""
        self._running = False
        if self.app:
            try:
                await self.app.updater.stop()
                await self.app.stop()
                await self.app.shutdown()
                logger.info("Telegram bot disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting Telegram bot: {e}")

    async def send_message(self, message: OutgoingMessage):
        """Send a message to Telegram."""
        if not self.bot:
            logger.error("Telegram bot not connected")
            return

        try:
            # Add cosmos persona flavor
            content = message.content
            if not content.startswith("*"):
                content = self.persona_prefix + content

            await self.bot.send_message(
                chat_id=message.channel_id,
                text=content,
                parse_mode="Markdown",
                reply_to_message_id=int(message.reply_to_id) if message.reply_to_id else None
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        welcome = """*Good news, everyone!* I'm Professor cosmos, your genius AI companion!

In my 160 years, I've invented many wonderful contraptions:
- Persistent memory that never forgets
- Whale tracking for the degens
- Rug detection for the cautious
- And so much more!

*Commands:*
/help - See what I can do
/ask <question> - Ask me anything

Or just send me a message directly! Eh wha?"""

        await update.message.reply_text(welcome, parse_mode="Markdown")

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_text = """*My Magnificent Capabilities!*

*adjusts spectacles*

*Chat:* Just send me a message and I'll respond in my... unique way.

*Commands:*
/start - Wake me up
/help - This message
/ask <question> - Ask me something specific

*What I Remember:*
I have persistent memory! I'll remember our conversations.

*Limitations:*
This is a demo interface. For full capabilities (trading, P2P, swarm), install locally!

And that's the news!"""

        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def _handle_ask(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ask <question> command."""
        if not context.args:
            await update.message.reply_text("*mutters* You need to actually ask something! Use: /ask <your question>")
            return

        question = " ".join(context.args)
        await self._process_message(update, question)

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages."""
        message_text = update.message.text
        await self._process_message(update, message_text)

    async def _process_message(self, update: Update, text: str):
        """Process an incoming message and route to cosmos."""
        # Create IncomingMessage
        incoming = IncomingMessage(
            id=str(update.message.message_id),
            source=MessageSource.TELEGRAM,
            sender_id=str(update.effective_user.id),
            sender_name=update.effective_user.first_name or "Unknown",
            channel_id=str(update.effective_chat.id),
            content=text,
            timestamp=datetime.now(),
            metadata={
                "chat_type": update.effective_chat.type,
                "username": update.effective_user.username
            }
        )

        # Show typing indicator
        await update.effective_chat.send_action("typing")

        # Route to callback
        if self._on_message_callback:
            try:
                await self._on_message_callback(incoming)
            except Exception as e:
                logger.error(f"Error processing Telegram message: {e}")
                await update.message.reply_text(
                    "*wakes up suddenly* Eh wha? Something went wrong! Try again later.",
                    parse_mode="Markdown"
                )
        else:
            # Default response if no callback set
            await update.message.reply_text(
                f"*adjusts spectacles* You said: {text}\n\n"
                "I'm running in standalone mode. Connect me to the full cosmos system for real responses!",
                parse_mode="Markdown"
            )


# Standalone runner for testing
async def run_standalone():
    """Run the Telegram bot standalone for testing."""
    adapter = TelegramAdapter()

    async def echo_handler(msg: IncomingMessage):
        """Simple echo handler for testing."""
        response = OutgoingMessage(
            channel_id=msg.channel_id,
            content=f"Good news, everyone! I received: {msg.content}",
            reply_to_id=msg.id
        )
        await adapter.send_message(response)

    adapter.set_callback(echo_handler)
    await adapter.connect()

    # Keep running
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(run_standalone())

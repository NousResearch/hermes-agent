"""LINE Messaging API platform adapter.

Supports two LINE Bot accounts (Platform.LINE and Platform.LINE_LYNX).
Uses Push API for all outbound messages (reply_token TTL too short for LLM).
Per-group persona routing via MessageEvent.auto_skill.
"""

import asyncio
import base64
import hashlib
import hmac
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDK availability guard
# ---------------------------------------------------------------------------

LINE_SDK_AVAILABLE = False
try:
    from linebot.v3.webhooks import (
        WebhookParser,
        MessageEvent as LineMessageEvent,
        InvalidSignatureError,
        TextMessageContent,
        ImageMessageContent,
        StickerMessageContent,
        AudioMessageContent,
    )
    LINE_SDK_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WEBHOOK_PORT = {
    "line": 18791,
    "line_lynx": 18792,
}
DEFAULT_WEBHOOK_PATH = {
    "line": "/webhook/line",
    "line_lynx": "/webhook/line/lynx",
}
LINE_PUSH_API_URL = "https://api.line.me/v2/bot/message/push"
LINE_CONTENT_API_URL = "https://api-data.line.me/v2/bot/message/{message_id}/content"
MAX_TEXT_LENGTH = 4900  # LINE limit is 5000; buffer of 100


def check_line_requirements() -> bool:
    """Check whether LINE SDK and credentials are available."""
    if not LINE_SDK_AVAILABLE:
        logger.warning("[line] line-bot-sdk not installed. Run: pip install line-bot-sdk")
        return False
    has_main = bool(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
    has_lynx = bool(os.getenv("LINE_LYNX_CHANNEL_ACCESS_TOKEN"))
    if not has_main and not has_lynx:
        logger.warning("[line] No LINE credentials found in environment")
        return False
    return True


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """Remove markdown formatting markers; preserve text content."""
    # Code blocks (``` ... ```)
    text = re.sub(r"```[a-zA-Z]*\n?(.*?)```", r"\1", text, flags=re.DOTALL)
    # Inline code
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Bold + italic (**...**)
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    # Italic (_..._)
    text = re.sub(r"_(.*?)_", r"\1", text)
    # Links [text](url)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Blockquotes
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def _split_text(text: str, max_len: int = MAX_TEXT_LENGTH) -> List[str]:
    """Split text into chunks <= max_len chars, splitting on whitespace boundaries."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while len(text) > max_len:
        split_pos = text.rfind(" ", 0, max_len)
        if split_pos <= 0:
            # No whitespace found — force split
            split_pos = max_len
        chunks.append(text[:split_pos].rstrip())
        text = text[split_pos:].lstrip()
    if text:
        chunks.append(text)
    return chunks

"""
Telegram Dispatcher
-------------------
Sends enriched alert briefings to Telegram channels or DMs.
"""

import os
import urllib.request
import urllib.parse
import json
from typing import Optional

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    "8764046749:AAHOX8PsdHiAFiUrzSD8LgDUFDd44zRBbCA"
)

HERMES_ALERTS_CHANNEL_ID = os.environ.get("HERMES_ALERTS_CHANNEL_ID", "-1003506715170")
DEFAULT_DM_CHAT_ID = "8500351481"  # stout DM (fallback)


# ─────────────────────────────────────────────────────────────────
# SEND MESSAGE
# ─────────────────────────────────────────────────────────────────

def send_message(
    text: str,
    chat_id: Optional[str] = None,
    token: Optional[str] = None,
    parse_mode: str = "Markdown",
    reply_to_message_id: Optional[int] = None,
) -> dict:
    """
    Send a Telegram message via the Bot API.

    Args:
        text: Message body. Markdown/Varkdown supported.
        chat_id: Target chat ID (channel or DM). Defaults to HERMES_ALERTS_CHANNEL_ID.
        token: Bot token. Defaults to TELEGRAM_BOT_TOKEN.
        parse_mode: "Markdown", "MarkdownV2", or "HTML".
        reply_to_message_id: Optional message ID to reply to.

    Returns:
        {"ok": bool, "message_id": int, "chat_id": str}
    """
    bot_token = token or TELEGRAM_BOT_TOKEN

    if not bot_token:
        print("[Telegram] ERROR: TELEGRAM_BOT_TOKEN not set", flush=True)
        return {"ok": False, "description": "TELEGRAM_BOT_TOKEN not set"}

    chat = chat_id or HERMES_ALERTS_CHANNEL_ID
    method = "sendMessage"

    payload = {
        "chat_id": chat,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    if reply_to_message_id:
        payload["reply_to_message_id"] = reply_to_message_id

    url = f"https://api.telegram.org/bot{bot_token}/{method}"

    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=15) as response:
            result = json.loads(response.read())

            if result.get("ok"):
                msg_id = result.get("result", {}).get("message_id")
                dest_chat = result.get("result", {}).get("chat", {}).get("id")
                print(f"[Telegram] Sent msg {msg_id} to {dest_chat}", flush=True)
                return {"ok": True, "message_id": msg_id, "chat_id": dest_chat}
            else:
                err = result.get("description", "Unknown error")
                print(f"[Telegram] API error: {err}", flush=True)
                return {"ok": False, "description": err}

    except Exception as e:
        print(f"[Telegram] Request failed: {e}", flush=True)
        return {"ok": False, "description": str(e)}


def send_briefing(
    briefing: str,
    alert_id: str,
    severity: str,
    device: str,
    chat_id: Optional[str] = None,
) -> dict:
    """
    Send an enriched alert briefing to Telegram.

    Wraps the briefing with a header (severity + device) and
    routes based on severity: P1→DM, else to channel.
    """
    # Emoji for severity
    severity_emoji = {
        "critical": "\u26a0\ufe0f",
        "high": "\u26a0\ufe0f",
        "warning": "\ud83d\udd0a",
        "low": "\u2139\ufe0f",
        "info": "\ud83d\udccc",
    }.get(severity.lower(), "\ud83d\udccc")

    header = (
        f"\u27a4\ufe0f *ENRICHED ALERT* {severity_emoji}\n"
        f"ID: `{alert_id}`  \u00b7  Device: `{device}`\n"
        f"\u2502"
    )

    text = f"{header}\n\n{briefing}"
    return send_message(text, chat_id=chat_id)


def set_webhook(url: str, token: Optional[str] = None) -> dict:
    """Set the Telegram bot webhook URL."""
    bot_token = token or TELEGRAM_BOT_TOKEN
    if not bot_token:
        return {"ok": False, "description": "TELEGRAM_BOT_TOKEN not set"}

    method = "setWebhook"
    payload = json.dumps({"url": url}).encode()

    try:
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{bot_token}/{method}",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read())
    except Exception as e:
        return {"ok": False, "description": str(e)}


def get_me(token: Optional[str] = None) -> dict:
    """Get bot info."""
    bot_token = token or TELEGRAM_BOT_TOKEN
    if not bot_token:
        return {"ok": False}
    try:
        url = f"https://api.telegram.org/bot{bot_token}/getMe"
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read())
    except Exception:
        return {"ok": False}

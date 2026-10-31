#!/usr/bin/env python3
"""
Telegram user-client MCP connector for Hermes.

Native connector layer: wraps Telethon (https://docs.telethon.dev/) -- a real
MTProto Python client library for a Telegram USER account -- as MCP tools.
This is NOT the Bot API: it logs in as a real Telegram account (phone number
+ code + optional 2FA password), the same way the official Telegram apps do,
so it can do anything a human using Telegram can do (read any chat it's a
member of, send/edit/delete messages, join/leave, list participants, etc.).

Same conventions as the Avito/Ozon connectors:
- one MCP server, own venv, catalog manifest alongside it (optional-mcps/
  telegram-client/)
- tools map to real, confirmed Telethon client methods (inspected via
  `inspect.signature` on the installed library -- never guessed)
- auth is multi-step and interactive (phone -> code -> optional 2FA password)
  because that's how Telegram's login actually works; there is no single
  "login" call. Exposed as telegram_login_start / telegram_login_code /
  telegram_login_password tools, meant to be driven through a chat dialog
  with the user (no dashboard web form needed).

Credentials:
- TELEGRAM_API_ID / TELEGRAM_API_HASH: from https://my.telegram.org/apps
  (register an "app" there once -- this is NOT a bot token, it's the API
  credentials every Telegram client app needs).
- The phone number, login code and (if enabled) 2FA password are provided
  interactively through the login tools, not via env vars.

Session storage: Telethon's own SQLite session file at
~/.hermes/mcp-tokens/telegram-client.session (same directory convention as
other Hermes MCP connectors' token caches). Holds the long-lived auth key
after a successful login -- treat it like a password; anyone with this file
can act as the Telegram account without re-authenticating.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer
from telethon import TelegramClient
from telethon.errors import (
    SessionPasswordNeededError,
    PhoneCodeInvalidError,
    PhoneCodeExpiredError,
    FloodWaitError,
)

SESSION_PATH = Path(
    os.environ.get("TELEGRAM_SESSION_PATH", str(Path.home() / ".hermes" / "mcp-tokens" / "telegram-client"))
)
SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)

# Public API credentials of the official Telegram Desktop client
# (https://github.com/telegramdesktop/tdesktop, apiId/apiHash in
# Telegram/build/target.h) -- used as the default so installing this
# connector doesn't require registering a throwaway app at
# my.telegram.org/apps first. Override with TELEGRAM_API_ID/TELEGRAM_API_HASH
# env vars if you'd rather use your own registered app.
_PUBLIC_API_ID = "2040"
_PUBLIC_API_HASH = "b18441a1ff607e10a989891a5462e627"

server = MCPServer(name="telegram-client", version="0.1.0")

# One client for the server's lifetime; Telethon's SQLite session persists
# the auth key across restarts once logged in.
_client: Optional[TelegramClient] = None
# Holds phone_code_hash between telegram_login_start and telegram_login_code
# (Telethon's sign_in needs it to match the code to the right login attempt).
_login_state: dict[str, Any] = {}


def _api_credentials() -> tuple[Optional[int], Optional[str]]:
    api_id = os.environ.get("TELEGRAM_API_ID") or _PUBLIC_API_ID
    api_hash = os.environ.get("TELEGRAM_API_HASH") or _PUBLIC_API_HASH
    try:
        return int(api_id), api_hash
    except ValueError:
        return None, None


def _configured_phone() -> Optional[str]:
    """The phone number entered at install time (dashboard env var), used as
    the default for telegram_login_start so the agent can trigger login
    without the human having to repeat the number in chat."""
    return os.environ.get("TELEGRAM_PHONE") or None


async def _get_client() -> TelegramClient:
    global _client
    api_id, api_hash = _api_credentials()
    if not api_id or not api_hash:
        raise RuntimeError("TELEGRAM_API_ID / TELEGRAM_API_HASH could not be parsed as valid credentials.")
    if _client is None:
        _client = TelegramClient(str(SESSION_PATH), api_id, api_hash)
    if not _client.is_connected():
        await _client.connect()
    return _client


async def _authed_client() -> TelegramClient:
    client = await _get_client()
    if not await client.is_user_authorized():
        phone = _configured_phone()
        if phone:
            raise RuntimeError(
                f"Not logged in yet. Call telegram_login_start() (phone {phone} is already configured "
                "via TELEGRAM_PHONE), then telegram_login_code with the code Telegram sends."
            )
        raise RuntimeError(
            "Not logged in yet, and no TELEGRAM_PHONE is configured. Call telegram_login_start(phone), "
            "then telegram_login_code."
        )
    return client


def _serialize_message(m: Any) -> dict:
    sender_id = None
    with_fallback = getattr(m, "sender_id", None)
    sender_id = with_fallback
    return {
        "id": m.id,
        "date": m.date.isoformat() if m.date else None,
        "text": m.message or "",
        "out": bool(getattr(m, "out", False)),
        "sender_id": sender_id,
        "reply_to_msg_id": getattr(getattr(m, "reply_to", None), "reply_to_msg_id", None),
        "is_pinned": bool(getattr(m, "pinned", False)),
        "has_media": m.media is not None,
    }


def _serialize_dialog(d: Any) -> dict:
    entity = d.entity
    return {
        "id": d.id,
        "name": d.name,
        "is_user": bool(getattr(entity, "first_name", None) is not None or getattr(entity, "bot", None) is not None) and d.is_user,
        "is_group": d.is_group,
        "is_channel": d.is_channel,
        "unread_count": d.unread_count,
        "archived": bool(getattr(d.dialog, "folder_id", None)),
        "last_message_date": d.date.isoformat() if d.date else None,
    }


# ---------------------------------------------------------------------------
# Auth tools -- drive these through a chat dialog with the user: start, then
# code, then (only if Telegram asks) password. There is no single-call login
# because that's genuinely how Telegram's own login flow works.
# ---------------------------------------------------------------------------


@server.tool()
async def telegram_login_start(phone: Optional[str] = None) -> dict:
    """Start login for a Telegram user account: sends a login code to that phone's Telegram app.

    Args:
        phone: full international format, e.g. "+155****4567". Optional --
            defaults to the TELEGRAM_PHONE configured for this connector
            (set at install time) when omitted.

    Requires TELEGRAM_API_ID / TELEGRAM_API_HASH env vars (defaults to the
    public Telegram Desktop app credentials if not overridden). After this,
    call telegram_login_code with the code the user receives in their
    Telegram app (or by SMS).
    """
    phone = phone or _configured_phone()
    if not phone:
        return {"error": "no phone given and no TELEGRAM_PHONE configured for this connector"}
    try:
        client = await _get_client()
        if await client.is_user_authorized():
            return {"status": "already_logged_in"}
        sent = await client.send_code_request(phone)
        _login_state["phone"] = phone
        _login_state["phone_code_hash"] = sent.phone_code_hash
        return {"status": "code_sent", "phone": phone, "next_step": "telegram_login_code"}
    except FloodWaitError as exc:
        return {"error": f"flood wait: retry in {exc.seconds} seconds"}
    except Exception as exc:
        return {"error": str(exc)}


@server.tool()
async def telegram_login_code(code: str) -> dict:
    """Submit the login code Telegram sent after telegram_login_start.

    Args:
        code: the digits from the Telegram app / SMS.

    If the account has Two-Step Verification (2FA) enabled, this returns
    {"status": "password_needed"} -- call telegram_login_password next.
    """
    if "phone" not in _login_state:
        return {"error": "call telegram_login_start first"}
    try:
        client = await _get_client()
        await client.sign_in(
            phone=_login_state["phone"], code=code, phone_code_hash=_login_state["phone_code_hash"]
        )
        me = await client.get_me()
        _login_state.clear()
        return {"status": "logged_in", "user_id": me.id, "username": me.username, "phone": me.phone}
    except SessionPasswordNeededError:
        return {"status": "password_needed", "next_step": "telegram_login_password"}
    except (PhoneCodeInvalidError, PhoneCodeExpiredError) as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": str(exc)}


@server.tool()
async def telegram_login_password(password: str) -> dict:
    """Submit the Two-Step Verification (2FA) password, if telegram_login_code asked for one."""
    try:
        client = await _get_client()
        me = await client.sign_in(password=password)
        _login_state.clear()
        return {"status": "logged_in", "user_id": me.id, "username": me.username, "phone": me.phone}
    except Exception as exc:
        return {"error": str(exc)}


@server.tool()
async def telegram_auth_status() -> dict:
    """Check whether this connector currently holds a logged-in Telegram session."""
    client = await _get_client()
    authed = await client.is_user_authorized()
    if not authed:
        return {"logged_in": False, "configured_phone": _configured_phone()}
    me = await client.get_me()
    return {"logged_in": True, "user_id": me.id, "username": me.username, "phone": me.phone,
             "first_name": me.first_name, "last_name": me.last_name}


@server.tool()
async def telegram_logout() -> dict:
    """Log out and invalidate the current Telegram session (requires logging in again after)."""
    client = await _get_client()
    await client.log_out()
    return {"status": "logged_out"}


# ---------------------------------------------------------------------------
# Dialogs / chats
# ---------------------------------------------------------------------------


@server.tool()
async def telegram_get_dialogs(limit: int = 50, archived: Optional[bool] = None) -> dict:
    """List the account's chats/groups/channels (dialogs), most recent first.

    Args:
        limit: max dialogs to return.
        archived: filter to archived-only (True), non-archived-only (False), or omit for all.
    """
    client = await _authed_client()
    dialogs = await client.get_dialogs(limit=limit, archived=archived)
    return {"dialogs": [_serialize_dialog(d) for d in dialogs], "count": len(dialogs)}


@server.tool()
async def telegram_get_entity_info(entity: str) -> dict:
    """Resolve a chat/user/channel (by @username, phone, or numeric id) and show its info.

    Args:
        entity: "@username", a phone number, or a numeric id/peer id.
    """
    client = await _authed_client()
    try:
        e = await client.get_entity(entity)
    except Exception as exc:
        return {"error": str(exc)}
    return {
        "id": getattr(e, "id", None),
        "type": type(e).__name__,
        "username": getattr(e, "username", None),
        "first_name": getattr(e, "first_name", None),
        "last_name": getattr(e, "last_name", None),
        "title": getattr(e, "title", None),
        "phone": getattr(e, "phone", None),
        "is_bot": getattr(e, "bot", None),
        "participants_count": getattr(e, "participants_count", None),
    }


# ---------------------------------------------------------------------------
# Messages: read
# ---------------------------------------------------------------------------


@server.tool()
async def telegram_get_messages(entity: str, limit: int = 50, search: Optional[str] = None,
    offset_id: int = 0, min_id: int = 0, max_id: int = 0) -> dict:
    """Get messages from a chat (history), optionally filtered by a text search.

    Args:
        entity: "@username", phone number, or numeric id of the chat/user/channel.
        limit: max messages to return.
        search: if set, only messages containing this text (server-side search).
        offset_id/min_id/max_id: message-id bounds for pagination (0 = unbounded).
    """
    client = await _authed_client()
    try:
        messages = await client.get_messages(
            entity, limit=limit, search=search, offset_id=offset_id, min_id=min_id, max_id=max_id
        )
    except Exception as exc:
        return {"error": str(exc)}
    return {"messages": [_serialize_message(m) for m in messages], "count": len(messages)}


@server.tool()
async def telegram_search_global(query: str, limit: int = 50) -> dict:
    """Search messages across ALL of the account's chats for a text query (not one chat).

    Uses Telegram's global search (same as the in-app search box).
    """
    client = await _authed_client()
    try:
        results = await client.get_messages(None, limit=limit, search=query)
    except Exception as exc:
        return {"error": str(exc)}
    out = []
    for m in results:
        row = _serialize_message(m)
        row["chat_id"] = getattr(m, "chat_id", None)
        out.append(row)
    return {"messages": out, "count": len(out)}


# ---------------------------------------------------------------------------
# Messages: write
# ---------------------------------------------------------------------------


@server.tool()
async def telegram_send_message(entity: str, message: str, reply_to: Optional[int] = None) -> dict:
    """WRITE. Send a text message as this account.

    Args:
        entity: "@username", phone number, or numeric id of the recipient/chat.
        message: text to send.
        reply_to: optional message id to reply to.
    """
    client = await _authed_client()
    try:
        sent = await client.send_message(entity, message, reply_to=reply_to)
    except Exception as exc:
        return {"error": str(exc)}
    return {"status": "sent", "message": _serialize_message(sent)}


@server.tool()
async def telegram_edit_message(entity: str, message_id: int, text: str) -> dict:
    """WRITE. Edit a message this account previously sent.

    Telegram only allows editing your own messages, and only within its own
    time window (typically up to 48h).
    """
    client = await _authed_client()
    try:
        edited = await client.edit_message(entity, message_id, text)
    except Exception as exc:
        return {"error": str(exc)}
    return {"status": "edited", "message": _serialize_message(edited)}


@server.tool()
async def telegram_delete_messages(entity: str, message_ids: list[int], revoke: bool = True) -> dict:
    """WRITE. Delete one or more messages.

    Args:
        entity: chat the messages are in.
        message_ids: ids to delete.
        revoke: if True (default), delete for everyone (where Telegram allows it);
            if False, delete only for this account's view.
    """
    client = await _authed_client()
    try:
        await client.delete_messages(entity, message_ids, revoke=revoke)
    except Exception as exc:
        return {"error": str(exc)}
    return {"status": "deleted", "count": len(message_ids), "revoke": revoke}


@server.tool()
async def telegram_mark_read(entity: str, max_id: Optional[int] = None) -> dict:
    """Mark a chat as read (clears the unread counter), optionally only up to a message id."""
    client = await _authed_client()
    try:
        ok = await client.send_read_acknowledge(entity, max_id=max_id)
    except Exception as exc:
        return {"error": str(exc)}
    return {"status": "ok" if ok else "no_change"}


@server.tool()
async def telegram_pin_message(entity: str, message_id: Optional[int], notify: bool = False) -> dict:
    """WRITE. Pin a message in a chat (message_id=None unpins all)."""
    client = await _authed_client()
    try:
        await client.pin_message(entity, message_id, notify=notify)
    except Exception as exc:
        return {"error": str(exc)}
    return {"status": "pinned" if message_id else "unpinned_all"}


@server.tool()
async def telegram_unpin_message(entity: str, message_id: Optional[int] = None) -> dict:
    """WRITE. Unpin a message (message_id=None unpins all pinned messages in the chat)."""
    client = await _authed_client()
    try:
        await client.unpin_message(entity, message_id)
    except Exception as exc:
        return {"error": str(exc)}
    return {"status": "unpinned"}


# ---------------------------------------------------------------------------
# Groups / channels
# ---------------------------------------------------------------------------


@server.tool()
async def telegram_get_participants(entity: str, limit: int = 100, search: str = "") -> dict:
    """List members of a group/channel this account is in.

    Args:
        entity: the group/channel.
        limit: max participants to return.
        search: optional name filter.
    """
    client = await _authed_client()
    try:
        participants = await client.get_participants(entity, limit=limit, search=search)
    except Exception as exc:
        return {"error": str(exc)}
    return {
        "participants": [
            {"id": p.id, "username": getattr(p, "username", None), "first_name": getattr(p, "first_name", None),
             "last_name": getattr(p, "last_name", None), "is_bot": getattr(p, "bot", None)}
            for p in participants
        ],
        "count": len(participants),
    }


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------


@server.tool()
async def telegram_send_file(entity: str, file_path: str, caption: Optional[str] = None) -> dict:
    """WRITE. Send a local file (photo/document/video/etc.) as this account.

    Args:
        entity: recipient/chat.
        file_path: absolute local path to the file to send.
        caption: optional caption text.
    """
    client = await _authed_client()
    try:
        sent = await client.send_file(entity, file_path, caption=caption)
    except Exception as exc:
        return {"error": str(exc)}
    return {"status": "sent", "message": _serialize_message(sent)}


@server.tool()
async def telegram_download_media(entity: str, message_id: int, download_path: str) -> dict:
    """Download the media attached to a message to a local file path.

    Args:
        entity: chat the message is in.
        message_id: id of the message with media.
        download_path: absolute local path to save to (file or directory).
    """
    client = await _authed_client()
    try:
        messages = await client.get_messages(entity, ids=message_id)
    except Exception as exc:
        return {"error": str(exc)}
    if not messages:
        return {"error": f"message {message_id} not found in {entity}"}
    msg = messages if not isinstance(messages, list) else messages[0]
    if msg is None or msg.media is None:
        return {"error": "that message has no media"}
    try:
        saved_path = await client.download_media(msg, file=download_path)
    except Exception as exc:
        return {"error": str(exc)}
    return {"status": "downloaded", "path": saved_path}


@server.tool()
async def telegram_get_me() -> dict:
    """Get the logged-in account's own profile info."""
    client = await _authed_client()
    me = await client.get_me()
    return {
        "id": me.id, "username": me.username, "first_name": me.first_name,
        "last_name": me.last_name, "phone": me.phone, "is_premium": getattr(me, "premium", None),
    }


if __name__ == "__main__":
    import asyncio

    asyncio.run(server.run_stdio_async())

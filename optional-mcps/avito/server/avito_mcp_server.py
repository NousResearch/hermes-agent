#!/usr/bin/env python3
"""
Avito API MCP connector for Hermes.

Native connector layer: wraps the `avito-api` PyPI package
(https://pypi.org/project/avito-api/) as MCP tools over real, confirmed
Avito endpoints (base URL https://api.avito.ru). Business logic ("which API
do I need for X") lives one layer up, in the `avito-api-*` Hermes skills —
this server only does authenticated HTTP I/O.

Confirmed from the avito-api library source (not guessed):
- OAuth token endpoint:        POST https://api.avito.ru/token
- Personal-auth authorize URL: https://avito.ru/oauth?response_type=code&client_id=...&scope=...
- Grant types: client_credentials (app auth), authorization_code (personal auth), refresh_token
- Scopes (enum): messenger:read, messenger:write, user_balance:read, job:write, job:cv,
  job:vacancy, job:applications, user_operations:read, user:read, autoload:reports,
  items:info, items:apply_vas, short_term_rent:read, short_term_rent:write, stats:read
- Messenger endpoints: /messenger/v2/accounts/{user_id}/chats, /messenger/v2/.../chats/{chat_id},
  /messenger/v3/.../chats/{chat_id}/messages/, /messenger/v1/.../chats/{chat_id}/messages (POST send),
  /messenger/v1/.../uploadImages, /messenger/v1/.../chats/{chat_id}/messages/image,
  /messenger/v1/.../chats/{chat_id}/read, /messenger/v1/.../chats/{chat_id}/messages/{message_id} (delete),
  /messenger/v1/.../getVoiceFiles, /messenger/v3/webhook, /messenger/v1/webhook/unsubscribe,
  /messenger/v1/subscriptions, /messenger/v2/accounts/{user_id}/blacklist
- Item endpoints: /core/v1/items, /core/v1/accounts/{user_id}/items/{item_id}/
- User endpoint: /core/v1/accounts/self

Everything else in Avito's 26-API catalog (Ads, CPA, Delivery, Autoload, ...)
is NOT wired here yet because no verified endpoint spec was available when
this scaffold was built (see the `avito-api-catalog` skill's Pitfalls on the
site's Qrator anti-bot block). Extend `TOOLS` below once those specs are
confirmed -- never guess a path or param here.
"""
from __future__ import annotations

import json
import os
import sys
import time
import webbrowser
import http.server
import threading
import urllib.parse
from pathlib import Path
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer

from avito_api.auth.client import SyncAuthClient
from avito_api.auth.models import Scope
from avito_api.messenger.client import SyncMessengerClient
from avito_api.messenger.models import BlacklistUser, BlacklistUserContext
from avito_api.item.client import SyncItemClient
from avito_api.user.client import SyncUserClient
from avito_api.exceptions import AvitoApiError, AuthError

# ---------------------------------------------------------------------------
# Token cache (mirrors Hermes's own ~/.hermes/mcp-tokens/<server>.json convention)
# ---------------------------------------------------------------------------
TOKEN_CACHE_PATH = Path(
    os.environ.get("AVITO_TOKEN_CACHE", str(Path.home() / ".hermes" / "mcp-tokens" / "avito.json"))
)


def _load_tokens() -> dict[str, Any]:
    if TOKEN_CACHE_PATH.exists():
        try:
            return json.loads(TOKEN_CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_tokens(data: dict[str, Any]) -> None:
    TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_CACHE_PATH.write_text(json.dumps(data, indent=2))
    try:
        os.chmod(TOKEN_CACHE_PATH, 0o600)
    except OSError:
        pass


def _client_id() -> str:
    v = os.environ.get("AVITO_CLIENT_ID")
    if not v:
        raise RuntimeError("AVITO_CLIENT_ID is not set (register an app at avito.ru/developers)")
    return v


def _client_secret() -> str:
    v = os.environ.get("AVITO_CLIENT_SECRET")
    if not v:
        raise RuntimeError("AVITO_CLIENT_SECRET is not set (register an app at avito.ru/developers)")
    return v


def _build_auth_client() -> SyncAuthClient:
    tokens = _load_tokens()
    return SyncAuthClient(
        client_id=_client_id(),
        client_secret=_client_secret(),
        access_token=tokens.get("access_token"),
        refresh_token=tokens.get("refresh_token"),
        token_expires_at=tokens.get("token_expires_at"),
    )


def _persist_from_client(client: SyncAuthClient) -> None:
    _save_tokens(
        {
            "access_token": client.access_token,
            "refresh_token": client.refresh_token,
            "token_expires_at": client.token_expires_at,
            "saved_at": int(time.time()),
        }
    )


def _authed_client(cls):
    """Build any Sync*Client of the library, pre-loaded with the cached token."""
    tokens = _load_tokens()
    if not tokens.get("access_token"):
        raise RuntimeError(
            "Not logged in to Avito yet. Call avito_login_app_auth (no browser) "
            "or avito_login_start / avito_login_finish (personal auth, browser) first."
        )
    return cls(
        client_id=_client_id(),
        client_secret=_client_secret(),
        access_token=tokens.get("access_token"),
        refresh_token=tokens.get("refresh_token"),
        token_expires_at=tokens.get("token_expires_at"),
    )


server = MCPServer(name="avito", version="0.1.0")

# ---------------------------------------------------------------------------
# Auth / login tools -- this is the "connector settings -> login" surface.
# Two modes:
#   1. App auth (client_credentials): no browser, works immediately once
#      AVITO_CLIENT_ID/SECRET are set. No "login" needed.
#   2. Personal auth (authorization_code): needs a one-time browser approval.
#      avito_login_start() returns a URL to open; the user approves in their
#      browser and is redirected to a localhost callback this tool is
#      listening on, or (no redirect_uri configured) shown a code to paste
#      back via avito_login_finish().
# ---------------------------------------------------------------------------


@server.tool()
def avito_login_app_auth() -> dict:
    """Log in with app-level auth (client_credentials grant, no browser).

    Requires AVITO_CLIENT_ID and AVITO_CLIENT_SECRET to be set in the
    environment. Use this when you only need endpoints that don't require a
    specific user's consent.
    """
    client = _build_auth_client()
    try:
        resp = client.get_token()
    except AuthError as e:
        return {"error": str(e)}
    _persist_from_client(client)
    return {"status": "logged_in", "mode": "app_auth", "expires_in": getattr(resp, "expires_in", None)}


@server.tool()
def avito_login_start(scopes: list[str], redirect_uri: Optional[str] = None) -> dict:
    """Start personal (authorization_code) login: returns a URL to open in a browser.

    Args:
        scopes: list of scope strings, e.g. ["messenger:read", "messenger:write",
            "user:read", "items:info"]. See avito_list_scopes for the full set.
        redirect_uri: optional; if your Avito app is registered with a redirect
            URI, pass it here so Avito redirects back with the code in the URL.
            If omitted, Avito shows the code directly on its confirmation page
            for you to copy.

    After the user approves in their browser, call avito_login_finish with the
    resulting `code`.
    """
    client = _build_auth_client()
    invalid = [s for s in scopes if s not in {x.value for x in Scope}]
    if invalid:
        return {"error": f"Unknown scope(s): {invalid}. Call avito_list_scopes for valid values."}
    url = client.get_authorization_url(client_id=_client_id(), scopes=scopes, redirect_uri=redirect_uri)
    return {
        "authorize_url": url,
        "instructions": (
            "Open this URL in a browser, log in to Avito, approve access, "
            "then call avito_login_finish with the 'code' you receive."
        ),
    }


@server.tool()
def avito_login_finish(code: str) -> dict:
    """Complete personal login: exchange the authorization code for tokens.

    Args:
        code: the `code` query param (or on-page code) Avito gave after
            approving the avito_login_start URL.
    """
    client = _build_auth_client()
    try:
        resp = client.get_token_by_authorization_code(code)
    except AuthError as e:
        return {"error": str(e)}
    _persist_from_client(client)
    return {"status": "logged_in", "mode": "personal_auth", "scope": getattr(resp, "scope", None)}


@server.tool()
def avito_auth_status() -> dict:
    """Show whether this connector currently holds a usable Avito token."""
    tokens = _load_tokens()
    if not tokens.get("access_token"):
        return {"logged_in": False}
    expires_at = tokens.get("token_expires_at")
    return {
        "logged_in": True,
        "token_expires_at": expires_at,
        "expired": bool(expires_at) and int(time.time()) >= expires_at,
        "has_refresh_token": bool(tokens.get("refresh_token")),
        "cache_path": str(TOKEN_CACHE_PATH),
    }


@server.tool()
def avito_logout() -> dict:
    """Delete the cached Avito token (forces a fresh login next time)."""
    if TOKEN_CACHE_PATH.exists():
        TOKEN_CACHE_PATH.unlink()
    return {"status": "logged_out"}


@server.tool()
def avito_list_scopes() -> dict:
    """List the OAuth scopes this connector knows about (for avito_login_start)."""
    return {"scopes": [s.value for s in Scope]}


# ---------------------------------------------------------------------------
# Messenger tools (confirmed endpoints: /messenger/v1,v2,v3/...)
# ---------------------------------------------------------------------------


@server.tool()
def avito_messenger_get_chats(user_id: int, item_ids: Optional[list[int]] = None, unread_only: bool = False) -> dict:
    """List the authenticated user's Avito Messenger chats.

    Args:
        user_id: Avito user (account) id that owns the chats.
        item_ids: optional filter to chats about specific listing ids.
        unread_only: only return chats with unread messages.
    """
    client = _authed_client(SyncMessengerClient)
    try:
        result = client.get_chats(user_id=user_id, item_ids=item_ids, unread_only=unread_only)
    except AvitoApiError as e:
        return {"error": str(e)}
    return result.model_dump() if hasattr(result, "model_dump") else result


@server.tool()
def avito_messenger_get_chat(user_id: int, chat_id: str) -> dict:
    """Get a single Avito Messenger chat by id."""
    client = _authed_client(SyncMessengerClient)
    try:
        result = client.get_chat(user_id=user_id, chat_id=chat_id)
    except AvitoApiError as e:
        return {"error": str(e)}
    return result.model_dump() if hasattr(result, "model_dump") else result


@server.tool()
def avito_messenger_get_messages(user_id: int, chat_id: str, limit: int = 100, offset: int = 0) -> dict:
    """Get messages in an Avito Messenger chat."""
    client = _authed_client(SyncMessengerClient)
    try:
        result = client.get_messages(user_id=user_id, chat_id=chat_id, limit=limit, offset=offset)
    except AvitoApiError as e:
        return {"error": str(e)}
    return result.model_dump() if hasattr(result, "model_dump") else result


@server.tool()
def avito_messenger_send_message(user_id: int, chat_id: str, text: str) -> dict:
    """Send a text message in an Avito Messenger chat."""
    client = _authed_client(SyncMessengerClient)
    try:
        result = client.send_message(user_id=user_id, chat_id=chat_id, text=text)
    except AvitoApiError as e:
        return {"error": str(e)}
    return result.model_dump() if hasattr(result, "model_dump") else result


@server.tool()
def avito_messenger_mark_read(user_id: int, chat_id: str) -> dict:
    """Mark an Avito Messenger chat as read."""
    client = _authed_client(SyncMessengerClient)
    try:
        result = client.mark_chat_as_read(user_id=user_id, chat_id=chat_id)
    except AvitoApiError as e:
        return {"error": str(e)}
    return result.model_dump() if hasattr(result, "model_dump") else result


@server.tool()
def avito_messenger_delete_message(user_id: int, chat_id: str, message_id: str) -> dict:
    """Delete a message in an Avito Messenger chat."""
    client = _authed_client(SyncMessengerClient)
    try:
        return client.delete_message(user_id=user_id, chat_id=chat_id, message_id=message_id)
    except AvitoApiError as e:
        return {"error": str(e)}


@server.tool()
def avito_messenger_blacklist_user(user_id: int, blocked_user_id: int, reason_id: int, item_id: Optional[int] = None) -> dict:
    """Add a user to the Messenger blacklist.

    Args:
        user_id: your Avito account id (the one blocking).
        blocked_user_id: the user id being blocked.
        reason_id: 1=spam, 2=fraud, 3=insult, 4=other (BlacklistReasonId).
        item_id: optional listing id this block relates to.
    """
    client = _authed_client(SyncMessengerClient)
    entry = BlacklistUser(user_id=blocked_user_id, context=BlacklistUserContext(item_id=item_id, reason_id=reason_id))
    try:
        return client.add_to_blacklist(user_id=user_id, blacklist_users=[entry])
    except AvitoApiError as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Item (listing) tools (confirmed endpoints: /core/v1/items, /core/v1/accounts/{id}/items/{id}/)
# ---------------------------------------------------------------------------


@server.tool()
def avito_item_get_items(status: Optional[str] = None, category: Optional[int] = None, page: int = 1, per_page: int = 25) -> dict:
    """List the authenticated user's Avito listings (items)."""
    client = _authed_client(SyncItemClient)
    try:
        result = client.get_items(status=status, category=category, page=page, per_page=per_page)
    except AvitoApiError as e:
        return {"error": str(e)}
    return result.model_dump() if hasattr(result, "model_dump") else result


@server.tool()
def avito_item_get_item(user_id: int, item_id: int) -> dict:
    """Get a single Avito listing (item) by id."""
    client = _authed_client(SyncItemClient)
    try:
        result = client.get_item(user_id=user_id, item_id=item_id)
    except AvitoApiError as e:
        return {"error": str(e)}
    return result.model_dump() if hasattr(result, "model_dump") else result


# ---------------------------------------------------------------------------
# User tools (confirmed endpoint: /core/v1/accounts/self)
# ---------------------------------------------------------------------------


@server.tool()
def avito_user_get_account() -> dict:
    """Get the authenticated Avito account's own profile info (id, name, etc.)."""
    client = _authed_client(SyncUserClient)
    try:
        result = client.get_account()
    except AvitoApiError as e:
        return {"error": str(e)}
    return result.model_dump() if hasattr(result, "model_dump") else result


if __name__ == "__main__":
    server.run_stdio_async() if False else None
    import asyncio

    asyncio.run(server.run_stdio_async())

"""``/setup-files`` in-chat OAuth setup flow for native attachment delivery.

Extracted from ``adapter.py``: ``GoogleChatAdapter._handle_setup_files_command``
delegates here. Logs under the adapter's pinned logger name.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("gateway.platforms.google_chat")

_NOT_CONFIGURED_TEXT = (
    "🔧 Native attachment delivery is **not configured**.\n"
    "**Step 1 (one-time, on the host):** create OAuth client credentials at "
    "https://console.cloud.google.com/apis/credentials → *Create credentials* → "
    "*OAuth client ID* → *Desktop app*. Download the JSON. Then on the host run:\n"
    "```\npython -m plugins.platforms.google_chat.oauth --client-secret /path/to/client_secret.json\n```\n"
    "**Step 2:** come back here and send `/setup-files start`."
)
_START_INSTRUCTIONS = (
    "1. Open this URL in your browser and authorize:\n{auth_url}\n\n"
    "2. After clicking *Allow*, your browser will fail to load "
    "`http://localhost:1/?...&code=...`. That's expected.\n\n"
    "3. Copy the entire failed URL from the browser's URL bar and paste it back here as: "
    "`/setup-files <PASTE_URL>` (or just the `code=...` value).\n\n"
    "Tip: the URL contains your access grant — keep it private."
)
_START_EXIT_TEXT = (
    "❌ Couldn't generate the OAuth URL. Check the gateway logs and verify the client_secret.json is valid."
)
_EXCHANGE_EXIT_TEXT = (
    "❌ Token exchange failed. The code may have expired or the URL is malformed. "
    "Send `/setup-files start` to get a fresh OAuth URL."
)
_REVOKE_EXIT_OUTPUT = "Revoke completed (some steps may have been skipped)."
_EXITED = object()  # _run_helper marker: helper called sys.exit but the step tolerates it


async def _run_captured(fn: Callable[..., Any], *args: Any) -> str:
    """Run ``fn`` in a thread with stdout captured (the oauth helpers print their output)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        await asyncio.to_thread(fn, *args)
    return buf.getvalue()


async def handle_setup_files_command(
    adapter: Any, chat_id: str, thread_id: Optional[str], raw_text: str,
    sender_email: Optional[str] = None) -> bool:
    """Run the in-chat OAuth setup flow. Returns True when the message was consumed.

    ``sender_email`` is the per-user OAuth key; ``None`` falls back to the legacy
    single-user token slot so pre-multi-user installs keep working.
    Subcommands: ``/setup-files`` (status), ``start`` (OAuth URL), ``revoke``,
    ``<CODE_OR_URL>`` (exchange). Requires client_secret.json on the host.
    """
    from . import oauth as oauth_helper

    # Same normalization as the token-path sanitizer so cache lookups stay consistent.
    sender_key = sender_email.strip().lower() if sender_email else None
    parts = raw_text.split(maxsplit=1)
    arg = parts[1].strip() if len(parts) > 1 else ""

    async def _reply(text: str) -> None:
        body: Dict[str, Any] = {"text": text}
        if thread_id:
            body["thread"] = {"name": thread_id}
        try:
            await adapter._create_message(chat_id, body)
        except Exception:
            logger.debug("[GoogleChat] /setup-files reply send failed", exc_info=True)

    async def _run_helper(step: str, exit_text: Optional[str], fn: Callable[..., Any], *args: Any):
        """Captured helper output; ``None`` after replying on failure. ``exit_text``
        is the reply on ``SystemExit`` (the helpers' failure signal); ``None``
        tolerates the exit and returns ``_EXITED``."""
        try:
            return await _run_captured(fn, *args)
        except SystemExit:
            if exit_text is None:
                return _EXITED
            await _reply(exit_text)
        except Exception as exc:
            logger.warning("[GoogleChat] /setup-files %s failed: %s", step, exc)
            await _reply(f"❌ Error{' revoking' if step == 'revoke' else ''}: {exc}")
        return None

    def _set_user_creds(creds: Any, api: Any) -> None:
        """Set (or evict, with ``None``) only the sender's slot: Bob revoking must not
        break Alice's per-user token nor the shared legacy fallback."""
        if not sender_key:
            adapter._user_credentials, adapter._user_chat_api = creds, api
        elif creds is None:
            adapter._user_creds_by_email.pop(sender_key, None)
            adapter._user_chat_api_by_email.pop(sender_key, None)
        else:
            adapter._user_creds_by_email[sender_key] = creds
            adapter._user_chat_api_by_email[sender_key] = api

    if not arg:
        client_secret_present = oauth_helper._client_secret_path().exists()
        token_path = oauth_helper._token_path(sender_key)
        creds = oauth_helper.load_user_credentials(sender_key) if token_path.exists() else None
        if creds is not None:
            who = sender_key or "shared (legacy)"
            await _reply(
                f"✅ Native attachment delivery is **active** for `{who}`.\n"
                f"Token: `{token_path}`\nSend `/setup-files revoke` to disable.")
        elif not client_secret_present:
            await _reply(_NOT_CONFIGURED_TEXT)
        else:
            await _reply(
                "🔧 Client credentials are stored but you haven't authorized yet. "
                "Send `/setup-files start` to begin."
            )
        return True

    if arg == "start":
        if not oauth_helper._client_secret_path().exists():
            await _reply(
                "⚠️ No client credentials stored for this profile. Send "
                "`/setup-files` (no args) for setup instructions."
            )
            return True
        output = await _run_helper("start", _START_EXIT_TEXT, oauth_helper.get_auth_url, sender_key)
        if output is not None:
            await _reply(_START_INSTRUCTIONS.format(auth_url=output.strip().splitlines()[-1]))
        return True

    if arg == "revoke":
        output = await _run_helper("revoke", None, oauth_helper.revoke, sender_key)
        if output is None:
            return True
        output = _REVOKE_EXIT_OUTPUT if output is _EXITED else (output.strip() or "Revoked.")
        _set_user_creds(None, None)
        await _reply(f"✅ Done.\n```\n{output}\n```")
        return True

    # Anything else is the auth code or the pasted failed-redirect URL.
    output = await _run_helper("exchange", _EXCHANGE_EXIT_TEXT, oauth_helper.exchange_auth_code, arg, sender_key)
    if output is None:
        return True
    # Re-load credentials so the next file send uses them without a gateway restart.
    try:
        new_creds = await asyncio.to_thread(oauth_helper.load_user_credentials, sender_key)
        if new_creds is not None:
            new_api = await asyncio.to_thread(lambda: oauth_helper.build_user_chat_service(new_creds))
            _set_user_creds(new_creds, new_api)
            await _reply("✅ Authorized! Native attachment delivery is now active. Try asking me to send you a PDF.")
            return True
    except Exception as exc:
        logger.warning("[GoogleChat] post-exchange creds load failed: %s", exc)
    await _reply(
        "⚠️ Token exchanged but the gateway couldn't load the new credentials in-memory. "
        f"Restart the gateway and the token at `{oauth_helper._token_path(sender_key)}` will be picked up.\n"
        f"Helper output:\n```\n{output.strip()}\n```")
    return True

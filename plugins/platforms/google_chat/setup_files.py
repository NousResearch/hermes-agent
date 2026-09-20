"""``/setup-files`` in-chat OAuth setup flow for native attachment delivery.

Extracted from ``adapter.py``: ``GoogleChatAdapter._handle_setup_files_command``
delegates here. Logs under the adapter's pinned logger name.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Optional

from agent.i18n import t

logger = logging.getLogger("gateway.platforms.google_chat")

# Reply copy lives in the catalog under platform.google_chat.setup_files.* and is resolved
# through ``t()`` at reply time (never at import) so the active language applies.
_K = "platform.google_chat.setup_files."
_EXITED = object()  # _run_helper marker: helper called sys.exit but the step tolerates it


async def _run_captured(fn: Callable[..., Any], *args: Any) -> str:
    """Run ``fn`` in a thread and return its string result.

    The oauth helpers used to be CLI-only (print + ``sys.exit``). Scraping
    their stdout with process-global ``redirect_stdout`` around
    ``asyncio.to_thread`` remapped every gateway thread's stdout for the
    duration, so a concurrent print could become the "OAuth URL"
    (``splitlines()[-1]``) and other platforms lost console output
    (#55769 class). Helpers now return the text they also print.
    """
    result = await asyncio.to_thread(fn, *args)
    return result if isinstance(result, str) else ""


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

    async def _run_helper(step: str, exit_key: Optional[str], fn: Callable[..., Any], *args: Any):
        """Captured helper output; ``None`` after replying on failure. ``exit_key``
        is the catalog key of the reply on ``SystemExit`` (the helpers' failure signal);
        ``None`` tolerates the exit and returns ``_EXITED``."""
        try:
            return await _run_captured(fn, *args)
        except SystemExit:
            if exit_key is None:
                return _EXITED
            await _reply(t(exit_key))
        except Exception as exc:
            logger.warning("[GoogleChat] /setup-files %s failed: %s", step, exc)
            await _reply(t(_K + ("revoke_error" if step == "revoke" else "helper_error"), error=str(exc)))
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
            who = sender_key or t(_K + "who_shared")
            await _reply(t(_K + "active", who=who, token_path=str(token_path)))
        elif not client_secret_present:
            await _reply(t(_K + "not_configured"))
        else:
            await _reply(t(_K + "not_authorized_yet"))
        return True

    if arg == "start":
        if not oauth_helper._client_secret_path().exists():
            await _reply(t(_K + "no_client_credentials"))
            return True
        output = await _run_helper("start", _K + "start_failed", oauth_helper.get_auth_url, sender_key)
        if output is not None:
            await _reply(t(_K + "start_instructions", auth_url=output.strip()))
        return True

    if arg == "revoke":
        output = await _run_helper("revoke", None, oauth_helper.revoke, sender_key)
        if output is None:
            return True
        output = t(_K + "revoke_completed") if output is _EXITED else (output.strip() or t(_K + "revoked"))
        _set_user_creds(None, None)
        await _reply(t(_K + "done_output", output=output))
        return True

    # Anything else is the auth code or the pasted failed-redirect URL.
    output = await _run_helper("exchange", _K + "exchange_failed", oauth_helper.exchange_auth_code, arg, sender_key)
    if output is None:
        return True
    # Re-load credentials so the next file send uses them without a gateway restart.
    try:
        new_creds = await asyncio.to_thread(oauth_helper.load_user_credentials, sender_key)
        if new_creds is not None:
            new_api = await asyncio.to_thread(lambda: oauth_helper.build_user_chat_service(new_creds))
            _set_user_creds(new_creds, new_api)
            await _reply(t(_K + "authorized"))
            return True
    except Exception as exc:
        logger.warning("[GoogleChat] post-exchange creds load failed: %s", exc)
    await _reply(t(_K + "exchanged_not_loaded", token_path=str(oauth_helper._token_path(sender_key)),
                   output=output.strip()))
    return True

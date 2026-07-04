"""Facebook Messenger (Meta Page DM) platform plugin for Hermes Agent.

Bundled platform plugin — registers via ``ctx.register_platform()`` with
zero core edits.  All protocol machinery (shared webhook server, HMAC
signature validation, Graph API sends) lives in :mod:`.meta_common`,
which the sibling ``instagram`` plugin shares: both surfaces belong to
one Meta app, receive events on the SAME webhook callback (routed by the
payload's top-level ``object`` field) and send through the same
``/me/messages`` endpoint with the same Page access token.

Setup summary (full guide: website/docs/user-guide/messaging/messenger.md):

1. Create a Meta app with the Messenger product and generate a Page
   access token for your Facebook Page.
2. In ``~/.hermes/.env`` set the shared credentials —
   ``META_PAGE_ACCESS_TOKEN``, ``META_APP_SECRET``, ``META_VERIFY_TOKEN``
   — plus ``MESSENGER_ENABLED=true``.
3. Expose the webhook (default ``127.0.0.1:8647/meta/webhook``) via a
   public HTTPS reverse proxy or tunnel and subscribe the Page webhook
   to ``messages`` with that callback URL.
4. Gate access with ``MESSENGER_ALLOWED_USERS`` (comma-separated PSIDs)
   or ``MESSENGER_ALLOW_ALL_USERS=true`` for a public customer bot.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from plugins.platforms.messenger.meta_common import (
    MESSENGER_MAX_CHARS,
    MESSENGER_SAFE_CHARS,
    MetaBaseAdapter,
    make_check_requirements,
    make_env_enablement,
    make_is_connected,
    make_standalone_send,
)


class MessengerAdapter(MetaBaseAdapter):
    """Facebook Messenger Page-DM adapter (Meta webhook + Graph API)."""

    PLATFORM_NAME = "messenger"
    ENV_PREFIX = "MESSENGER"
    CHAT_LABEL = "Messenger DM"
    MAX_MESSAGE_LENGTH = MESSENGER_MAX_CHARS
    SAFE_CHUNK_CHARS = MESSENGER_SAFE_CHARS


check_requirements = make_check_requirements("MESSENGER")
is_connected = make_is_connected("MESSENGER")
validate_config = is_connected
_env_enablement = make_env_enablement("MESSENGER")
_standalone_send = make_standalone_send(MessengerAdapter)

PLATFORM_HINT = (
    "You are on Facebook Messenger talking with the user in a direct "
    "message. Messenger does NOT render markdown — asterisks and hashes "
    "show literally, so reply in plain text and keep replies short and "
    "conversational. Messages are chunked at 2000 characters. Publicly "
    "reachable image URLs in your MEDIA/image tool results are sent as "
    "native photos; local file attachments cannot be uploaded — share a "
    "link instead."
)


def interactive_setup() -> None:
    """Minimal stdin wizard for ``hermes gateway setup`` → Messenger."""
    print()
    print("Facebook Messenger (Meta) setup")
    print("-------------------------------")
    print("1. Create a Meta app with the Messenger product:")
    print("   https://developers.facebook.com/apps/")
    print("2. Generate a Page access token for your Facebook Page and copy")
    print("   the app secret from App settings > Basic.")
    print("3. Pick any random string as your webhook verify token.")
    print()

    try:
        from hermes_cli.config import get_env_value, save_env_value
    except ImportError:
        print(
            "hermes_cli.config not available; set META_* and MESSENGER_* "
            "vars manually in ~/.hermes/.env"
        )
        return

    def _prompt(var: str, prompt: str, *, secret: bool = False) -> None:
        existing = get_env_value(var)
        suffix = " [keep current]" if existing else ""
        try:
            if secret:
                from hermes_cli.secret_prompt import masked_secret_prompt
                value = masked_secret_prompt(f"{prompt}{suffix}: ")
            else:
                value = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if value:
            save_env_value(var, value)

    _prompt("META_PAGE_ACCESS_TOKEN", "Page access token", secret=True)
    _prompt("META_APP_SECRET", "App secret", secret=True)
    _prompt("META_VERIFY_TOKEN", "Webhook verify token", secret=True)
    _prompt(
        "MESSENGER_ALLOWED_USERS",
        "Allowed PSIDs (comma-separated; blank=skip — set "
        "MESSENGER_ALLOW_ALL_USERS=true for a public bot)",
    )
    save_env_value("MESSENGER_ENABLED", "true")
    print()
    print("Done. Expose http://127.0.0.1:8647/meta/webhook via public HTTPS")
    print("(reverse proxy or tunnel), then in the Meta developer portal set")
    print("that URL + your verify token as the Messenger webhook callback and")
    print("subscribe the Page to 'messages'.")


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="messenger",
        label="Messenger (Meta)",
        adapter_factory=lambda cfg: MessengerAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=[
            "MESSENGER_ENABLED",
            "META_PAGE_ACCESS_TOKEN",
            "META_APP_SECRET",
            "META_VERIFY_TOKEN",
        ],
        install_hint="pip install aiohttp",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="MESSENGER_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="MESSENGER_ALLOWED_USERS",
        allow_all_env="MESSENGER_ALLOW_ALL_USERS",
        max_message_length=MESSENGER_MAX_CHARS,
        emoji="💠",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=PLATFORM_HINT,
    )

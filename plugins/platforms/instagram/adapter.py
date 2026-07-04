"""Instagram DM platform plugin for Hermes Agent.

Bundled platform plugin — registers via ``ctx.register_platform()`` with
zero core edits.  Instagram Direct Messages are the second surface of a
Meta app: an Instagram professional account is linked to a Facebook
Page, events arrive on the SAME webhook callback as Messenger (payloads
carry ``object: "instagram"``), and sends go through the same Graph API
``/me/messages`` endpoint with the same Page access token.

All protocol machinery therefore lives in
:mod:`plugins.platforms.messenger.meta_common`, shared with the
``messenger`` plugin the way the two WhatsApp core adapters share
``gateway/platforms/whatsapp_common.py``.  When both plugins are enabled
they attach to ONE shared webhook listener and events are routed to the
right adapter by the payload's ``object`` field, so sessions are
recorded under the correct platform (``instagram`` vs ``messenger``).

Setup summary (full guide: website/docs/user-guide/messaging/instagram.md):

1. Link an Instagram professional account to the Facebook Page of your
   Meta app and add the Instagram product to the app.
2. Use the same shared credentials as Messenger in ``~/.hermes/.env``
   (``META_PAGE_ACCESS_TOKEN``, ``META_APP_SECRET``,
   ``META_VERIFY_TOKEN``) plus ``INSTAGRAM_ENABLED=true``.
3. Subscribe the app's Instagram webhook to ``messages`` — the callback
   URL is the same one Messenger uses (default
   ``127.0.0.1:8647/meta/webhook`` behind your HTTPS proxy/tunnel).
4. Gate access with ``INSTAGRAM_ALLOWED_USERS`` (comma-separated
   Instagram-scoped IDs) or ``INSTAGRAM_ALLOW_ALL_USERS=true`` for a
   public customer bot.
"""

from __future__ import annotations

from plugins.platforms.messenger.meta_common import (
    INSTAGRAM_MAX_CHARS,
    INSTAGRAM_SAFE_CHARS,
    MetaBaseAdapter,
    make_check_requirements,
    make_env_enablement,
    make_is_connected,
    make_standalone_send,
)


class InstagramAdapter(MetaBaseAdapter):
    """Instagram DM adapter (Meta webhook + Graph API)."""

    PLATFORM_NAME = "instagram"
    ENV_PREFIX = "INSTAGRAM"
    CHAT_LABEL = "Instagram DM"
    MAX_MESSAGE_LENGTH = INSTAGRAM_MAX_CHARS
    SAFE_CHUNK_CHARS = INSTAGRAM_SAFE_CHARS


check_requirements = make_check_requirements("INSTAGRAM")
is_connected = make_is_connected("INSTAGRAM")
validate_config = is_connected
_env_enablement = make_env_enablement("INSTAGRAM")
_standalone_send = make_standalone_send(InstagramAdapter)

PLATFORM_HINT = (
    "You are on Instagram Direct Messages talking with the user. "
    "Instagram does NOT render markdown — reply in plain text and keep "
    "replies short and conversational; messages are hard-capped at 1000 "
    "characters and chunked accordingly. Publicly reachable image URLs "
    "in your MEDIA/image tool results are sent as native photos; local "
    "file attachments cannot be uploaded — share a link instead."
)


def interactive_setup() -> None:
    """Minimal stdin wizard for ``hermes gateway setup`` → Instagram."""
    print()
    print("Instagram DM (Meta) setup")
    print("-------------------------")
    print("Instagram shares its Meta app credentials with Messenger:")
    print("1. Link an Instagram professional account to your Facebook Page.")
    print("2. If you already configured Messenger, the META_* values below")
    print("   are the same — just press Enter to keep them.")
    print()

    try:
        from hermes_cli.config import get_env_value, save_env_value
    except ImportError:
        print(
            "hermes_cli.config not available; set META_* and INSTAGRAM_* "
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
        "INSTAGRAM_ALLOWED_USERS",
        "Allowed Instagram-scoped IDs (comma-separated; blank=skip — set "
        "INSTAGRAM_ALLOW_ALL_USERS=true for a public bot)",
    )
    save_env_value("INSTAGRAM_ENABLED", "true")
    print()
    print("Done. Subscribe the app's Instagram webhook to 'messages' in the")
    print("Meta developer portal — the callback URL and verify token are the")
    print("same ones Messenger uses (default /meta/webhook on port 8647).")


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="instagram",
        label="Instagram DM (Meta)",
        adapter_factory=lambda cfg: InstagramAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=[
            "INSTAGRAM_ENABLED",
            "META_PAGE_ACCESS_TOKEN",
            "META_APP_SECRET",
            "META_VERIFY_TOKEN",
        ],
        install_hint="pip install aiohttp",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="INSTAGRAM_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="INSTAGRAM_ALLOWED_USERS",
        allow_all_env="INSTAGRAM_ALLOW_ALL_USERS",
        max_message_length=INSTAGRAM_MAX_CHARS,
        emoji="📸",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=PLATFORM_HINT,
    )

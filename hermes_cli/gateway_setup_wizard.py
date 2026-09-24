"""Gateway platform setup wizard (hermes gateway setup): platform registry/status table, per-platform setup prompts, service offer.

Extracted from ``hermes_cli/gateway.py``. Bodies read facade helpers through ``_gw()`` (late
binding on ``hermes_cli.gateway``) so the seams tests and callers patch on the facade keep
intercepting the moved code.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from hermes_cli.setup import print_success  # def-time binding (table value)
from hermes_cli.setup import print_warning  # def-time binding (table value)


def _gw():
    from hermes_cli import gateway  # late: the facade imports this module
    return gateway


# Built-in per-platform setup config (env vars, instructions, prompts). Telegram, WhatsApp, Email,
# SMS, etc. live in plugins/platforms/<name>/ and are discovered via the platform registry.
_PLATFORMS = [
    {
        "key": "mattermost", "label": "Mattermost", "emoji": "💬", "token_var": "MATTERMOST_TOKEN",
        "setup_instructions": [
            "1. In Mattermost: Integrations → Bot Accounts → Add Bot Account",
            "   (System Console → Integrations → Bot Accounts must be enabled)",
            "2. Give it a username (e.g. hermes) and copy the bot token",
            "3. Works with any self-hosted Mattermost instance — enter your server URL",
            "4. To find your user ID: click your avatar (top-left) → Profile",
            "   Your user ID is displayed there — click it to copy.",
            "   ⚠ This is NOT your username — it's a 26-character alphanumeric ID.",
            "5. To get a channel ID: click the channel name → View Info → copy the ID",
        ],
        "vars": [
            {"name": "MATTERMOST_URL", "prompt": "Server URL (e.g. https://mm.example.com)",
             "password": False, "help": "Your Mattermost server URL. Works with any self-hosted instance."},
            {"name": "MATTERMOST_TOKEN", "prompt": "Bot token", "password": True,
             "help": "Paste the bot token from step 2 above."},
            {"name": "MATTERMOST_ALLOWED_USERS", "prompt": "Allowed user IDs (comma-separated)",
             "password": False, "is_allowlist": True, "help": "Your Mattermost user ID from step 4 above."},
            {"name": "MATTERMOST_HOME_CHANNEL",
             "prompt": "Home channel ID (for cron/notification delivery, or empty to set later with /set-home)",
             "password": False, "help": "Channel ID where Hermes delivers cron results and notifications."},
            {"name": "MATTERMOST_REPLY_MODE",
             "prompt": "Reply mode — 'off' for flat messages, 'thread' for threaded replies (default: off)",
             "password": False,
             "help": "off = flat channel messages, thread = replies nest under your message."},
        ],
    },
    {"key": "signal", "label": "Signal", "emoji": "📡", "token_var": "SIGNAL_HTTP_URL"},
    {"key": "weixin", "label": "Weixin / WeChat", "emoji": "💬", "token_var": "WEIXIN_ACCOUNT_ID"},
    {
        "key": "bluebubbles", "label": "BlueBubbles (iMessage)",
        "emoji": "💬", "token_var": "BLUEBUBBLES_SERVER_URL",
        "setup_instructions": [
            "1. Install BlueBubbles on a Mac that will act as your iMessage server:",
            "   https://bluebubbles.app/",
            "2. Complete the BlueBubbles setup wizard — sign in with your Apple ID",
            "3. In BlueBubbles Settings → API, note the Server URL and password",
            "4. The server URL is typically http://<your-mac-ip>:1234",
            "5. Hermes connects via the BlueBubbles REST API and receives",
            "   incoming messages via a local webhook",
            "6. To authorize users, use DM pairing: hermes pairing generate bluebubbles",
            "   Share the code — the user sends it via iMessage to get approved",
        ],
        "vars": [
            {"name": "BLUEBUBBLES_SERVER_URL",
             "prompt": "BlueBubbles server URL (e.g. http://192.168.1.10:1234)", "password": False,
             "help": "The URL shown in BlueBubbles Settings → API."},
            {"name": "BLUEBUBBLES_PASSWORD", "prompt": "BlueBubbles server password", "password": True,
             "help": "The password shown in BlueBubbles Settings → API."},
            {"name": "BLUEBUBBLES_ALLOWED_USERS",
             "prompt": "Pre-authorized phone numbers or iMessage IDs (comma-separated, or leave empty for DM pairing)",
             "password": False, "is_allowlist": True,
             "help": "Optional — pre-authorize specific users. Leave empty to use DM pairing instead (recommended)."},
            {"name": "BLUEBUBBLES_HOME_CHANNEL",
             "prompt": "Home channel (phone number or iMessage ID for cron/notifications, or empty)",
             "password": False,
             "help": "Phone number or Apple ID to deliver cron results and notifications to."},
        ],
    },
    {
        "key": "qqbot", "label": "QQ Bot", "emoji": "🐧", "token_var": "QQ_APP_ID",
        "setup_instructions": [
            "1. Register a QQ Bot application at q.qq.com",
            "2. Note your App ID and App Secret from the application page",
            "3. Enable the required intents (C2C, Group, Guild messages)",
            "4. Configure sandbox or publish the bot",
        ],
        "vars": [
            {"name": "QQ_APP_ID", "prompt": "QQ Bot App ID", "password": False,
             "help": "Your QQ Bot App ID from q.qq.com."},
            {"name": "QQ_CLIENT_SECRET", "prompt": "QQ Bot App Secret", "password": True,
             "help": "Your QQ Bot App Secret from q.qq.com."},
            {"name": "QQ_ALLOWED_USERS",
             "prompt": "Allowed user OpenIDs (comma-separated, leave empty for open access)",
             "password": False, "is_allowlist": True,
             "help": "Optional — restrict DM access to specific user OpenIDs."},
            {"name": "QQBOT_HOME_CHANNEL",
             "prompt": "Home channel (user/group OpenID for cron delivery, or empty)", "password": False,
             "help": "OpenID to deliver cron results and notifications to."},
        ],
    },
    {
        "key": "yuanbao", "label": "Yuanbao", "emoji": "💎", "token_var": "YUANBAO_APP_ID",
        "setup_instructions": [
            "1. Download the Yuanbao app from https://yuanbao.tencent.com/",
            "2. In the app, go to PAI → My Bot and create a new bot",
            "3. After the bot is created, copy the App ID and App Secret",
            "4. Enter them below and Hermes will connect automatically over WebSocket",
        ],
        "vars": [
            {"name": "YUANBAO_APP_ID", "prompt": "App ID", "password": False,
             "help": "The App ID from your Yuanbao IM Bot credentials."},
            {"name": "YUANBAO_APP_SECRET", "prompt": "App Secret", "password": True,
             "help": "The App Secret (used for HMAC signing) from your Yuanbao IM Bot."},
        ],
    },
]


def _all_platforms() -> list[dict]:
    """Built-in ``_PLATFORMS`` plus registry plugin platforms (same dict shape, source in
    ``_registry_entry``). Plugins are discovered here (idempotent) so the setup menu works without a
    running gateway; user-installed ones still need ``plugins.enabled`` (untrusted code). Matrix is
    hidden on Windows: python-olm has no wheel or native build (use WSL)."""
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
    except Exception as e:
        _gw().logger.debug("plugin discovery failed during platform enumeration: %s", e)

    hide_matrix = sys.platform == "win32"
    platforms = [dict(p) for p in _gw()._PLATFORMS if not (hide_matrix and p.get("key") == "matrix")]
    by_key = {p["key"]: p for p in platforms}

    try:
        from gateway.platform_registry import platform_registry
    except Exception:
        return platforms

    for entry in platform_registry.all_entries():
        if entry.name in by_key or (hide_matrix and entry.name == "matrix"):
            continue
        platforms.append({
            "key": entry.name, "label": entry.label, "emoji": entry.emoji,
            "token_var": entry.required_env[0] if entry.required_env else "",
            "install_hint": entry.install_hint, "_registry_entry": entry,
        })
    return platforms


def _platform_status(platform: dict) -> str:
    """Plain-text status string; uncolored because ANSI codes break curses menu width math."""
    entry = platform.get("_registry_entry")
    if entry is not None:
        # Prefer is_connected (env + config.yaml) over check_fn (a coarse deps gate). Never fall back
        # to check_fn when is_connected returned False, or "SDK installed" would override "no token".
        try:
            if entry.is_connected is not None:
                from gateway.config import PlatformConfig
                configured = bool(entry.is_connected(PlatformConfig(enabled=True)))
            else:
                configured = bool(entry.check_fn())
        except Exception:
            configured = False
        return "configured" if configured else "not configured"

    token_var = platform.get("token_var", "")
    if not token_var:
        return "not configured"
    # Built-ins needing a second credential to count as fully configured.
    second_var = {"signal": "SIGNAL_ACCOUNT", "weixin": "WEIXIN_TOKEN"}.get(platform.get("key"))
    present = [bool(_gw().get_env_value(v)) for v in (token_var, second_var) if v]
    if all(present):
        return "configured"
    return "partially configured" if any(present) else "not configured"


def _set_platform_unauthorized_dm_behavior(platform_key: str, behavior: str) -> None:
    """Persist a platform-specific unauthorized-DM policy in config.yaml."""
    _gw().write_platform_config_field(platform_key, "unauthorized_dm_behavior", behavior, raw=True)


def _print_setup_header(title: str) -> None:
    print()
    print(_gw().color(f"  ─── {title} Setup ───", _gw().Colors.CYAN))


def _confirm_reconfigure(label: str, *env_vars: str) -> bool:
    """False when ``label`` is already configured (all ``env_vars`` set) and the user declines."""
    if all(_gw().get_env_value(v) for v in env_vars):
        print()
        _gw().print_success(f"{label} is already configured.")
        return _gw().prompt_yes_no(f"  Reconfigure {label}?", False)
    return True


def _offer_home_channel(home_var: str, user_id: str, what: str) -> None:
    """Offer to persist ``user_id`` as ``home_var`` (e.g. "your Telegram user ID")."""
    if _gw().prompt_yes_no(f"  Use {what} ({user_id}) as the home channel?", True):
        _gw().save_env_value(home_var, user_id)
        _gw().print_success(f"  Home channel set to {user_id}")


def _save_env_values(**values: str) -> None:
    for name, value in values.items():
        _gw().save_env_value(name, value)


def _prompt_csv(prompt_text: str, default: str) -> str:
    """Comma-separated ID prompt with whitespace stripped."""
    return _gw().prompt(prompt_text, default, password=False).replace(" ", "")


# (default index, *choices) for the no-allowlist access prompt, keyed by is_email.
_UNAUTHORIZED_ACCESS_CHOICES = {
    True: (3,
        "Enable open access (any email sender can message the bot)",
        "Use DM pairing (unknown email senders receive a pairing code)",
        "Politely decline unknown senders (one-time message, then silence)",
        "Keep unknown senders silent"),
    False: (1,
        "Enable open access (anyone can message the bot)",
        "Use DM pairing (unknown users request access, you approve with 'hermes pairing approve')",
        "Politely decline unknown senders (one-time message, then silence)",
        "Skip for now (bot will deny all users until configured)"),
}


def _prompt_unauthorized_access(platform_key: str) -> None:
    """No allowlist was given — ask open access vs DM pairing vs decline vs skip/silent, and persist."""
    is_email = platform_key == "email"
    print()
    default_idx, *access_choices = _UNAUTHORIZED_ACCESS_CHOICES[is_email]
    access_idx = _gw().prompt_choice("  How should unauthorized users be handled?", access_choices, default_idx)
    if access_idx == 0:
        _gw().save_env_value("EMAIL_ALLOW_ALL_USERS" if is_email else "GATEWAY_ALLOW_ALL_USERS", "true")
        _gw().print_warning("  Open access enabled — anyone can use your bot!")
    elif access_idx == 1:
        if is_email:
            _set_platform_unauthorized_dm_behavior("email", "pair")
        _gw().print_success("  DM pairing mode — users will receive a code to request access.")
        _gw().print_info("  Approve with: hermes pairing approve <platform> <code>")
    elif access_idx == 2:
        _set_platform_unauthorized_dm_behavior(platform_key, "decline")
        _gw().print_success("  Unknown senders get one polite decline, then silence (unauthorized_dm_behavior: decline).")
    elif is_email:
        _gw().print_success("  Unknown email senders will be ignored.")
    else:
        _gw().print_info("  Skipped — configure later with 'hermes gateway setup'")


def _telegram_auto_setup(token_var: str) -> tuple[bool, object]:
    """Offer the managed-bot QR flow. Returns (token_saved, owner_user_id)."""
    print()
    _gw()._print_info_lines(
        "  Telegram can be configured automatically with a managed bot:",
        "  [1] Automatic (scan QR → confirm in Telegram → done)", "  [2] Manual BotFather token",
    )
    if _gw().prompt("  Choice [1/2]", default="1").strip() != "1":
        return False, None
    try:
        from hermes_cli.telegram_managed_bot import (
            auto_setup_telegram_bot_result, is_valid_telegram_bot_token,
        )
    except ImportError:
        _gw().print_warning("  Automatic setup is unavailable in this install.")
        return False, None
    result = auto_setup_telegram_bot_result()
    if result and is_valid_telegram_bot_token(result.token):
        _gw().save_env_value(token_var, result.token)
        _gw().print_success("  Saved TELEGRAM_BOT_TOKEN")
        return True, result.owner_user_id
    if result:
        _gw().print_warning("  Automatic setup returned an invalid Telegram token.")
    print()
    _gw().print_info("  Falling back to manual setup...")
    return False, None


def _clean_discord_ids(cleaned: str) -> str:
    """Strip common Discord prefixes (user:123, <@123>, <@!123>) from a comma-separated list."""
    parts = []
    for uid in cleaned.split(","):
        uid = uid.strip()
        if uid.startswith("<@") and uid.endswith(">"):
            uid = uid.lstrip("<@!").rstrip(">")
        if uid.lower().startswith("user:"):
            uid = uid[5:]
        if uid:
            parts.append(uid)
    return ",".join(parts)


def _prompt_allowlist_var(var: dict, platform_key: str, auto_owner_user_id) -> str | None:
    """Allowlist prompt for one var; returns the saved value or None (open-access prompt shown)."""
    if "TELEGRAM" in var["name"] and auto_owner_user_id:
        detected_id = str(auto_owner_user_id)
        _gw().print_success(f"  Detected your Telegram user ID: {detected_id}")
        if _gw().prompt_yes_no("  Allow this Telegram account to use the bot?", True):
            extra = _gw().prompt("  Additional allowed user IDs (comma-separated, optional)", password=False)
            ids = [detected_id]
            for uid in extra.replace(" ", "").split(","):
                if uid and uid not in ids:
                    ids.append(uid)
            cleaned = ",".join(ids)
            _gw().save_env_value(var["name"], cleaned)
            _gw().print_success("  Saved — only these users can interact with the bot.")
            return cleaned

    _gw()._print_info_lines(
        "  The gateway DENIES all users by default for security.",
        "  Enter user IDs to create an allowlist, or leave empty",
        "  and you'll be asked about open access next.",
    )
    value = _gw().prompt(f"  {var['prompt']}", password=False)
    if not value:
        _prompt_unauthorized_access(platform_key)
        return None
    cleaned = value.replace(" ", "")
    if "DISCORD" in var["name"]:
        cleaned = _clean_discord_ids(cleaned)
    _gw().save_env_value(var["name"], cleaned)
    _gw().print_success("  Saved — only these users can interact with the bot.")
    return cleaned


def _setup_standard_platform(platform: dict):
    """Interactive setup for Telegram, Discord, or Slack."""
    from hermes_cli.setup_hidden_env import is_setup_hidden_env as _is_setup_hidden_env
    emoji, label, token_var = platform["emoji"], platform["label"], platform["token_var"]
    _print_setup_header(f"{emoji} {label}")

    instructions = platform.get("setup_instructions")
    if instructions:
        print()
        _gw()._print_info_lines(*(f"  {line}" for line in instructions))

    if not _confirm_reconfigure(label, token_var):
        return

    auto_token_saved, auto_owner_user_id = False, None
    if platform.get("key") == "telegram":
        auto_token_saved, auto_owner_user_id = _telegram_auto_setup(token_var)

    allowed_val_set = None  # Track if user set an allowlist (for home channel offer)

    # Skip knobs the setup forms hide (home channel, reply mode, proxy...): they're self-configuring.
    setup_vars = [
        v for v in platform["vars"]
        if v["name"] == token_var or v.get("is_allowlist") or not _is_setup_hidden_env(v["name"])
    ]

    for var in setup_vars:
        print()
        _gw().print_info(f"  {var['help']}")
        existing = _gw().get_env_value(var["name"])
        if existing and var["name"] != token_var:
            _gw().print_info(f"  Current: {existing}")

        if auto_token_saved and var["name"] == token_var:
            _gw().print_info("  Token saved by automatic setup.")
            continue

        if var.get("is_allowlist"):
            saved = _prompt_allowlist_var(var, platform.get("key"), auto_owner_user_id)
            if saved is not None:
                allowed_val_set = saved
            continue

        value = _gw().prompt(f"  {var['prompt']}", password=var.get("password", False))
        if value:
            _gw().save_env_value(var["name"], value)
            _gw().print_success(f"  Saved {var['name']}")
        elif var["name"] == token_var:
            _gw().print_warning(f"  Skipped — {label} won't work without this.")
            return
        else:
            _gw().print_info("  Skipped (can configure later)")

    # Offer the first allowlisted user ID as home channel when none is set (Telegram DMs).
    home_var = f"{label.upper()}_HOME_CHANNEL"
    home_val = _gw().get_env_value(home_var)
    if allowed_val_set and not home_val and label == "Telegram":
        first_id = allowed_val_set.split(",")[0].strip()
        if first_id:
            _offer_home_channel(home_var, first_id, "your user ID")

    print()
    _gw().print_success(f"{emoji} {label} configured!")


# Weixin DM policy by menu index (index 2 = allowlist is prompted separately).
_WEIXIN_DM_POLICIES = {
    0: ("pairing", "false", print_success, "  DM pairing enabled."),
    1: ("open", "true", print_warning, "  Open DM access enabled for Weixin."),
    3: ("disabled", "false", print_warning, "  Direct messages disabled."),
}


_WEIXIN_GROUP_NOTE = (
    "  Note: QR login connects an iLink bot identity (e.g. ...@im.bot), not a",
    "  scriptable personal WeChat account. Ordinary WeChat groups typically cannot",
    "  invite an @im.bot identity, and iLink does not deliver ordinary-group events",
    "  to most bot accounts. The settings below only apply when iLink actually",
    "  delivers group events for your account type — otherwise DM remains the only",
    "  working channel regardless of this choice.",
)


def _setup_weixin():
    """Interactive setup for Weixin / WeChat personal accounts."""
    _print_setup_header("💬 Weixin / WeChat")
    print()
    _gw()._print_info_lines(
        "  1. Hermes will open Tencent iLink QR login in this terminal.",
        "  2. Use WeChat to scan and confirm the QR code.",
        "  3. Hermes will store the returned account_id/token in ~/.hermes/.env.",
        "  4. This adapter supports native text, image, video, and document delivery.",
    )

    if not _confirm_reconfigure("Weixin", "WEIXIN_ACCOUNT_ID", "WEIXIN_TOKEN"):
        return

    try:
        from gateway.platforms.weixin import check_weixin_requirements, qr_login
    except Exception as exc:
        _gw().print_error(f"  Weixin adapter import failed: {exc}")
        _gw().print_info("  Install gateway dependencies first, then retry.")
        return

    if not check_weixin_requirements():
        _gw().print_error("  Missing dependencies: Weixin needs aiohttp and cryptography.")
        _gw().print_info("  Install them, then rerun `hermes gateway setup`.")
        return

    print()
    if not _gw().prompt_yes_no("  Start QR login now?", True):
        _gw().print_info("  Cancelled.")
        return

    try:
        credentials = _gw().asyncio.run(qr_login(str(_gw().get_hermes_home())))
    except KeyboardInterrupt:
        print()
        _gw().print_warning("  Weixin setup cancelled.")
        return
    except Exception as exc:
        _gw().print_error(f"  QR login failed: {exc}")
        return

    if not credentials:
        _gw().print_warning("  QR login did not complete.")
        return

    account_id = credentials.get("account_id", "")
    user_id = credentials.get("user_id", "")
    _gw().save_env_value("WEIXIN_ACCOUNT_ID", account_id)
    _gw().save_env_value("WEIXIN_TOKEN", credentials.get("token", ""))
    if credentials.get("base_url", ""):
        _gw().save_env_value("WEIXIN_BASE_URL", credentials.get("base_url", ""))
    _gw().save_env_value(
        "WEIXIN_CDN_BASE_URL", _gw().get_env_value("WEIXIN_CDN_BASE_URL") or "https://novac2c.cdn.weixin.qq.com/c2c"
    )

    print()
    access_choices = [
        "Use DM pairing approval (recommended)", "Allow all direct messages", "Only allow listed user IDs",
        "Disable direct messages",
    ]
    access_idx = _gw().prompt_choice("  How should direct messages be authorized?", access_choices, 0)
    if access_idx == 2:
        allowlist = _prompt_csv("  Allowed Weixin user IDs (comma-separated)", user_id or "")
        _save_env_values(
            WEIXIN_DM_POLICY="allowlist", WEIXIN_ALLOW_ALL_USERS="false", WEIXIN_ALLOWED_USERS=allowlist
        )
        _gw().print_success("  Weixin allowlist saved.")
    else:
        policy, allow_all, emit, message = _WEIXIN_DM_POLICIES.get(access_idx, _WEIXIN_DM_POLICIES[3])
        _save_env_values(WEIXIN_DM_POLICY=policy, WEIXIN_ALLOW_ALL_USERS=allow_all, WEIXIN_ALLOWED_USERS="")
        emit(message)
        if access_idx == 0:
            _gw().print_info(
                "  Unknown DM users can request access and you approve them with `hermes pairing approve`."
            )

    print()
    _gw()._print_info_lines(*_WEIXIN_GROUP_NOTE)
    group_choices = [
        "Disable group chats (recommended)", "Allow all group chats", "Only allow listed group chat IDs",
    ]
    group_idx = _gw().prompt_choice("  How should group chats be handled?", group_choices, 0)
    if group_idx == 0:
        _save_env_values(WEIXIN_GROUP_POLICY="disabled", WEIXIN_GROUP_ALLOWED_USERS="")
        _gw().print_info("  Group chats disabled.")
    elif group_idx == 1:
        _save_env_values(WEIXIN_GROUP_POLICY="open", WEIXIN_GROUP_ALLOWED_USERS="")
        _gw().print_warning("  All group chats enabled (only takes effect if iLink delivers group events).")
    else:
        allow_groups = _prompt_csv("  Allowed group chat IDs (comma-separated, not member user IDs)", "")
        _save_env_values(WEIXIN_GROUP_POLICY="allowlist", WEIXIN_GROUP_ALLOWED_USERS=allow_groups)
        _gw().print_success("  Group allowlist saved (only takes effect if iLink delivers group events).")

    if user_id:
        print()
        _offer_home_channel("WEIXIN_HOME_CHANNEL", user_id, "your Weixin user ID")

    print()
    _gw().print_success("Weixin configured!")
    _gw().print_info(f"  Account ID: {account_id}")
    if user_id:
        _gw().print_info(f"  User ID: {user_id}")


def _setup_qqbot():
    """Interactive setup for QQ Bot — scan-to-configure or manual credentials."""
    _print_setup_header("🐧 QQ Bot")

    if not _confirm_reconfigure("QQ Bot", "QQ_APP_ID", "QQ_CLIENT_SECRET"):
        return

    print()
    method_choices = ["Scan QR code to add bot automatically (recommended)", "Enter existing App ID and App Secret manually"]
    credentials = None
    if _gw().prompt_choice("  How would you like to set up QQ Bot?", method_choices, 0) == 0:
        try:
            from gateway.platforms.qqbot import qr_register
            credentials = qr_register()
        except KeyboardInterrupt:
            print()
            _gw().print_warning("  QQ Bot setup cancelled.")
            return
        if not credentials:
            _gw().print_info("  QR setup did not complete. Continuing with manual input.")

    if not credentials:
        print()
        _gw()._print_info_lines(
            "  Go to https://q.qq.com to register a QQ Bot application.",
            "  Note your App ID and App Secret from the application page.",
        )
        print()
        app_id = _gw().prompt("  App ID", password=False)
        if not app_id:
            _gw().print_warning("  Skipped — QQ Bot won't work without an App ID.")
            return
        app_secret = _gw().prompt("  App Secret", password=True)
        if not app_secret:
            _gw().print_warning("  Skipped — QQ Bot won't work without an App Secret.")
            return
        credentials = {"app_id": app_id.strip(), "client_secret": app_secret.strip(), "user_openid": ""}

    _gw().save_env_value("QQ_APP_ID", credentials["app_id"])
    _gw().save_env_value("QQ_CLIENT_SECRET", credentials["client_secret"])

    user_openid = credentials.get("user_openid", "")

    print()
    access_choices = ["Use DM pairing approval (recommended)", "Allow all direct messages", "Only allow listed user OpenIDs"]
    access_idx = _gw().prompt_choice("  How should direct messages be authorized?", access_choices, 0)
    if access_idx == 0:
        _gw().save_env_value("QQ_ALLOW_ALL_USERS", "false")
        allowed = ""
        if user_openid:
            print()
            if _gw().prompt_yes_no(f"  Add yourself ({user_openid}) to the allow list?", True):
                allowed = user_openid
                _gw().print_success(f"  Allow list set to {user_openid}")
        _gw().save_env_value("QQ_ALLOWED_USERS", allowed)
        _gw().print_success("  DM pairing enabled.")
        _gw().print_info("  Unknown users can request access; approve with `hermes pairing approve`.")
    elif access_idx == 1:
        _save_env_values(QQ_ALLOW_ALL_USERS="true", QQ_ALLOWED_USERS="")
        _gw().print_warning("  Open DM access enabled for QQ Bot.")
    else:
        allowlist = _prompt_csv("  Allowed user OpenIDs (comma-separated)", user_openid or "")
        _save_env_values(QQ_ALLOW_ALL_USERS="false", QQ_ALLOWED_USERS=allowlist)
        _gw().print_success("  Allowlist saved.")

    print()
    if user_openid:
        _offer_home_channel("QQBOT_HOME_CHANNEL", user_openid, "your QQ user ID")
    else:
        home_channel = _gw().prompt("  Home channel OpenID (for cron/notifications, or empty)", password=False)
        if home_channel:
            _gw().save_env_value("QQBOT_HOME_CHANNEL", home_channel.strip())
            _gw().print_success(f"  Home channel set to {home_channel.strip()}")

    print()
    _gw().print_success("🐧 QQ Bot configured!")
    _gw().print_info(f"  App ID: {credentials['app_id']}")


def _signal_line_input(prompt_text: str) -> str | None:
    """``line_input`` for the Signal wizard; None (after printing the cancel line) on EOF/Ctrl+C."""
    try:
        return _gw().line_input(prompt_text).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Setup cancelled.")
        return None


def _setup_signal():
    """Interactive setup for Signal messenger."""
    _print_setup_header("📡 Signal")

    existing_url = _gw().get_env_value("SIGNAL_HTTP_URL")
    existing_account = _gw().get_env_value("SIGNAL_ACCOUNT")
    if not _confirm_reconfigure("Signal", "SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT"):
        return

    print()
    if shutil.which("signal-cli"):
        _gw().print_success("signal-cli found on PATH.")
    else:
        _gw().print_warning("signal-cli not found on PATH.")
        _gw()._print_info_lines(
            "  Signal requires signal-cli running as an HTTP daemon.", "  Install options:",
            "    Linux:  download from https://github.com/AsamK/signal-cli/releases",
            "    macOS:  brew install signal-cli", "    Docker: bbernhard/signal-cli-rest-api",
        )
        print()
        _gw()._print_info_lines(
            "  After installing, link your account and start the daemon:",
            '    signal-cli link -n "HermesAgent"',
            "    signal-cli --account +YOURNUMBER daemon --http 127.0.0.1:8080",
        )
        print()

    print()
    _gw().print_info("  Enter the URL where signal-cli HTTP daemon is running.")
    default_url = existing_url or "http://127.0.0.1:8080"
    url = _signal_line_input(f"  HTTP URL [{default_url}]: ")
    if url is None:
        return
    url = url or default_url

    _gw().print_info("  Testing connection...")
    try:
        import httpx
        resp = httpx.get(f"{url.rstrip('/')}/api/v1/check", timeout=10.0)
        if resp.status_code == 200:
            _gw().print_success("  signal-cli daemon is reachable!")
        else:
            _gw().print_warning(f"  signal-cli responded with status {resp.status_code}.")
            if not _gw().prompt_yes_no("  Continue anyway?", False):
                return
    except Exception as e:
        _gw().print_warning(f"  Could not reach signal-cli at {url}: {e}")
        if not _gw().prompt_yes_no("  Save this URL anyway? (you can start signal-cli later)", True):
            return

    _gw().save_env_value("SIGNAL_HTTP_URL", url)

    print()
    _gw()._print_info_lines("  Enter your Signal account phone number in E.164 format.", "  Example: +15551234567")
    default_account = existing_account or ""
    account = _signal_line_input(f"  Account number{f' [{default_account}]' if default_account else ''}: ")
    if account is None:
        return
    account = account or default_account
    if not account:
        _gw().print_error("  Account number is required.")
        return

    _gw().save_env_value("SIGNAL_ACCOUNT", account)

    print()
    _gw()._print_info_lines(
        "  The gateway DENIES all users by default for security.",
        "  Enter phone numbers or UUIDs of allowed users (comma-separated).",
    )
    default_allowed = _gw().get_env_value("SIGNAL_ALLOWED_USERS") or account
    allowed = _signal_line_input(f"  Allowed users [{default_allowed}]: ")
    if allowed is None:
        return
    _gw().save_env_value("SIGNAL_ALLOWED_USERS", allowed or default_allowed)

    print()
    if _gw().prompt_yes_no("  Enable group messaging? (disabled by default for security)", False):
        print()
        _gw().print_info("  Enter group IDs to allow, or * for all groups.")
        existing_groups = _gw().get_env_value("SIGNAL_GROUP_ALLOWED_USERS") or ""
        groups = _signal_line_input(f"  Group IDs [{existing_groups or '*'}]: ")
        if groups is None:
            return
        _gw().save_env_value("SIGNAL_GROUP_ALLOWED_USERS", groups or existing_groups or "*")

    print()
    _gw().print_success("Signal configured!")
    _gw()._print_info_lines(
        f"  URL: {url}", f"  Account: {account}", "  DM auth: via SIGNAL_ALLOWED_USERS + DM pairing",
        f"  Groups: {'enabled' if _gw().get_env_value('SIGNAL_GROUP_ALLOWED_USERS') else 'disabled'}",
    )


def _builtin_setup_fn(key: str):
    """Resolve a built-in platform's setup function; late-bound to dodge the hermes_cli.setup cycle."""
    from hermes_cli import setup as _s
    return {
        # telegram/discord/slack/whatsapp/dingtalk/feishu/wecom setup_fns come from their plugins.
        "bluebubbles": _gw().setup_platforms._setup_bluebubbles,
        "webhooks": _gw().setup_platforms._setup_webhooks,
        "signal": _setup_signal,
        "weixin": _setup_weixin,
        "qqbot": _setup_qqbot,
    }.get(key)


def _configure_platform(platform: dict) -> None:
    """Plugin ``setup_fn`` -> built-in by key -> ``_setup_standard_platform`` (``vars``) -> env-var hint.
    Bundled plugins auto-load; user plugins must already be in ``plugins.enabled``."""
    entry = platform.get("_registry_entry")
    fn = entry.setup_fn if entry is not None else None
    if fn is None:
        fn = _builtin_setup_fn(platform["key"])
    if fn is not None:
        fn()
        return
    if platform.get("vars"):
        _gw()._setup_standard_platform(platform)
        return

    label = platform.get("label", platform["key"])
    _print_setup_header(f"{platform.get('emoji', '🔌')} {label}")
    required = entry.required_env if entry else []
    if required:
        _gw().print_info(f"  Set these env vars in ~/.hermes/.env: {', '.join(required)}")
    else:
        _gw().print_info(f"  Configure {label} in config.yaml under gateway.platforms.{platform['key']}")
    if platform.get("install_hint"):
        _gw().print_info(f"  {platform['install_hint']}")


def _wizard_offer_service_action(action: str, question: str, failed_label: str, **kwargs) -> None:
    """Wizard start/restart prompt; prints remediation instead when system scope would need root."""
    if _gw().supports_systemd_services() and _gw()._system_scope_wizard_would_need_root():
        _gw()._print_system_scope_remediation(action)
    elif _gw().prompt_yes_no(question, True):
        _gw()._setup_service_action(action, failed_label=failed_label, **kwargs)


def _setup_service_action(
    action: str, *, failed_label: str, windows: bool = True, system: bool = False
) -> None:
    """Run a wizard service start/restart, printing remediation instead of raising. ``windows=False``
    skips Windows (pre-platform status block never offers it); ``system`` is a fresh install's scope."""
    try:
        backend = _gw()._service_backend(windows=windows)
        if backend is not None:
            _gw()._service_call(backend, action, None if action == "restart" else system)
        elif action == "restart" and windows:
            _gw().stop_profile_gateway()
            _gw().print_info("Start manually: hermes gateway")
    except _gw().UserSystemdUnavailableError as e:
        _gw().print_error(f"  {failed_label} — user systemd not reachable:")
        _gw()._print_indented(str(e))
    except _gw().SystemScopeRequiresRootError as e:
        # Defense in depth: the wizard's root pre-check should have caught this.
        _gw().print_error(f"  {failed_label}: {e}")
        _gw()._print_system_scope_remediation(action)
    except subprocess.CalledProcessError as e:
        _gw().print_error(f"  {failed_label}: {e}")


_WIZARD_BANNER = (
    "┌─────────────────────────────────────────────────────────┐",
    "│             ☤ Gateway Setup                            │",
    "├─────────────────────────────────────────────────────────┤",
    "│  Configure messaging platforms and the gateway service. │",
    "│  Press Ctrl+C at any time to exit.                     │",
    "└─────────────────────────────────────────────────────────┘",
)


_WIZARD_BACKEND_LABELS = {"systemd": "systemd", "launchd": "launchd", "windows": "Scheduled Task"}


# Post-setup guidance when no service backend applies, keyed by the fallthrough reason.
_WIZARD_NO_SERVICE_LINES = {
    "wsl": (
        "  WSL detected but systemd is not running.", "  Run in foreground: hermes gateway run",
        "  For persistence:   tmux new -s hermes 'hermes gateway run'",
        "  To enable systemd: add systemd=true to /etc/wsl.conf, then 'wsl --shutdown'",
    ),
    "termux": (
        "  Termux does not use systemd/launchd services.", "  Run in foreground: hermes gateway run",
        "  Or start it manually in the background (best effort): nohup hermes gateway run >{home}/logs/gateway.log 2>&1 &",
    ),
    "unsupported": (
        "  Service install not supported on this platform.", "  Run in foreground: hermes gateway run",
    ),
}


def _wizard_service_status_block() -> None:
    """Pre-platform service status: warnings, then offer to start an installed-but-stopped service."""
    print()
    service_installed = _gw()._is_service_installed()
    service_running = _gw()._is_service_running()

    if _gw().supports_systemd_services() and _gw().has_conflicting_systemd_units():
        _gw().print_systemd_scope_conflict_warning()
        print()

    if _gw().supports_systemd_services() and _gw().has_legacy_hermes_units():
        _gw().print_legacy_unit_warning()
        print()

    if service_installed and service_running:
        _gw().print_success("Gateway service is installed and running.")
    elif service_installed:
        _gw().print_warning("Gateway service is installed but not running.")
        _wizard_offer_service_action("start", "  Start it now?", "Failed to start", windows=False)
    else:
        _gw().print_info("Gateway service is not installed yet.")
        _gw().print_info("You'll be offered to install it after configuring platforms.")


def _wizard_platform_loop() -> None:
    while True:
        print()
        _gw().print_header("Messaging Platforms")

        platforms = _gw()._all_platforms()
        menu_items = [f"{p['emoji']} {p['label']}  ({_platform_status(p)})" for p in platforms] + ["Done"]
        choice = _gw().prompt_choice("Select a platform to configure:", menu_items, len(menu_items) - 1)
        if choice == len(platforms):
            break
        _gw()._configure_platform(platforms[choice])


def _wizard_install_service(backend: str) -> None:
    """Fresh install from the wizard: ask start-now / start-on-login once, install, then start.

    The Windows installer owns its start decision (Scheduled Task and Startup-folder
    paths start the gateway themselves when start_now is true, and a UAC hand-off
    installs and starts in the elevated child), so the wizard forwards the answers
    and returns without a second start. Each install-intent question is asked exactly
    once per setup run."""
    wsl_note = " (note: services may not survive WSL restarts)" if _gw().is_wsl() else ""
    start_now = _gw().prompt_yes_no("  Start the gateway now?", True)
    start_on_login = _gw().prompt_yes_no(
        f"  Start the gateway automatically on login/boot as a {_WIZARD_BACKEND_LABELS[backend]} service?"
        f"{wsl_note}",
        True,
    )
    if not (start_now or start_on_login):
        _gw().print_info("  Skipped start and auto-start setup.")
        _gw().print_info("  You can install later: hermes gateway install")
        if _gw().supports_systemd_services():
            _gw().print_info("  Or as a boot-time service: sudo hermes gateway install --system")
        _gw().print_info("  Or run in foreground:  hermes gateway run")
        return
    try:
        installed_scope, did_install = None, True
        if backend == "systemd":
            installed_scope, did_install = _gw().install_linux_gateway_from_setup(
                force=False, enable_on_startup=start_on_login
            )
        elif backend == "launchd":
            _gw().launchd_install(force=False)
        else:
            _gw()._gw_windows().install(force=False, start_now=start_now, start_on_login=start_on_login)
            return
        print()
        if did_install and start_now:
            _gw()._setup_service_action("start", failed_label="Start failed", system=installed_scope == "system")
    except subprocess.CalledProcessError as e:
        _gw().print_error(f"  Install failed: {e}")
        _gw().print_info("  You can try manually: hermes gateway install")


def _wizard_post_setup() -> None:
    """Offer to install/start/restart the gateway once at least one platform has progress."""
    print()
    print(_gw().color("─" * 58, _gw().Colors.DIM))
    if _gw()._served_profile_needs_no_service():
        return
    service_installed = _gw()._is_service_installed()
    service_running = _gw()._is_service_running()

    if service_running:
        _wizard_offer_service_action("restart", "  Restart the gateway to pick up changes?", "Restart failed")
    elif service_installed:
        _wizard_offer_service_action("start", "  Start the gateway service?", "Start failed")
    else:
        print()
        backend = _gw()._service_backend()
        if backend is not None:
            _gw()._wizard_install_service(backend)
            return
        if _gw().is_wsl():
            reason, home = "wsl", ""
        elif _gw().is_termux():
            from hermes_constants import display_hermes_home as _dhh
            reason, home = "termux", _dhh()
        else:
            reason, home = "unsupported", ""
        _gw()._print_info_lines(*(line.format(home=home) for line in _WIZARD_NO_SERVICE_LINES[reason]))


def gateway_setup():
    """Interactive setup for messaging platforms + gateway service."""
    if _gw().is_managed():
        _gw().managed_error("run gateway setup")
        return

    print()
    for banner_line in _WIZARD_BANNER:
        print(_gw().color(banner_line, _gw().Colors.MAGENTA))

    _wizard_service_status_block()
    _wizard_platform_loop()

    # Meaningful progress on any platform; ``_platform_status`` already handles plugin dual states.
    def _is_progress(status: str) -> bool:
        s = status.lower()
        return not (s == "not configured" or s.startswith("partially") or s.startswith("plugin disabled"))

    if any(_is_progress(_gw()._platform_status(p)) for p in _gw()._all_platforms()):
        _gw()._wizard_post_setup()
    else:
        print()
        _gw().print_info("No platforms configured. Run 'hermes gateway setup' when ready.")

    print()

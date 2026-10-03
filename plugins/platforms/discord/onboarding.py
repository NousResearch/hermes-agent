"""Discord onboarding: check a bot token against Discord and walk the user through setup.

The Developer Portal is where Discord setup goes wrong: a token copied from the wrong page, the
Message Content Intent left off (the bot connects and sees empty messages), an invite URL built
by hand, and Developer Mode just to learn your own user ID. ``GET /applications/@me`` with the
bot token answers all of it at once — the token works, which intents are on, how many servers the
bot is in, and who owns the application — so the CLI wizard and the dashboard/Desktop check route
both read :func:`check_bot_token` instead of asking the user to verify those by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlencode

# Every permission the adapter exercises (text, threads, reactions, voice). Named so the integer
# in the invite link and the docs can be re-derived instead of trusted.
INVITE_PERMISSION_BITS = {
    "Add Reactions": 6,
    "View Channels": 10,
    "Send Messages": 11,
    "Embed Links": 14,
    "Attach Files": 15,
    "Read Message History": 16,
    "Use External Emojis": 18,
    "Connect": 20,
    "Speak": 21,
    "Create Public Threads": 35,
    "Send Messages in Threads": 38,
}
INVITE_PERMISSIONS = sum(1 << bit for bit in INVITE_PERMISSION_BITS.values())
_TEAM_MEMBER_ACCEPTED = 2


@dataclass(frozen=True)
class DiscordBotCheck:
    app_id: str
    bot_name: str
    # (user_id, username): the application owner, or every accepted member of the owning team.
    owners: tuple[tuple[str, str], ...]
    message_content: bool
    server_members: bool
    server_count: Optional[int]

    @property
    def invite_url(self) -> str:
        # integration_type=0 is a server install, so Discord skips the "add to server or to my
        # apps?" question for applications that also allow user installs.
        query = urlencode({"client_id": self.app_id, "scope": "bot applications.commands",
                           "permissions": INVITE_PERMISSIONS, "integration_type": 0})
        return f"https://discord.com/oauth2/authorize?{query}"

    @property
    def bot_settings_url(self) -> str:
        return f"https://discord.com/developers/applications/{self.app_id}/bot"


def _owners(app: dict) -> tuple[tuple[str, str], ...]:
    team = app.get("team")
    if isinstance(team, dict):
        return tuple(
            (str(m["user"]["id"]), str(m["user"].get("username") or m["user"]["id"]))
            for m in team.get("members") or []
            if m.get("membership_state") == _TEAM_MEMBER_ACCEPTED and m.get("user", {}).get("id")
        )
    owner = app.get("owner") or {}
    return ((str(owner["id"]), str(owner.get("username") or owner["id"])),) if owner.get("id") else ()


def check_bot_token(token: str) -> DiscordBotCheck:
    """Read the application behind ``token``. Raises :class:`DiscordAPIError` (401 = bad token)
    or ``OSError`` when Discord is unreachable."""
    from tools.discord_tool import _FLAGS_GUILD_MEMBERS, _FLAGS_MESSAGE_CONTENT, _discord_request
    app = _discord_request("GET", "/applications/@me", token.strip(), timeout=10)
    flags = int(app.get("flags") or 0)
    bot = app.get("bot") or {}
    count = app.get("approximate_guild_count")
    return DiscordBotCheck(
        app_id=str(app["id"]),
        bot_name=str(bot.get("username") or app.get("name") or app["id"]),
        owners=_owners(app),
        message_content=bool(flags & _FLAGS_MESSAGE_CONTENT),
        server_members=bool(flags & _FLAGS_GUILD_MEMBERS),
        server_count=count if isinstance(count, int) else None,
    )


# ── CLI wizard (hermes setup / hermes gateway setup) ─────────────────────────


def _clean_discord_user_ids(raw: str) -> list:
    """Strip common Discord mention prefixes from a comma-separated ID string."""
    cleaned = []
    for uid in raw.replace(" ", "").split(","):
        uid = uid.strip()
        if uid.startswith("<@") and uid.endswith(">"):
            uid = uid.lstrip("<@!").rstrip(">")
        if uid.lower().startswith("user:"):
            uid = uid[5:]
        if uid:
            cleaned.append(uid)
    return cleaned


def _discord_token_shape_error(token: str) -> Optional[str]:
    """Reject a Discord bot token that is really the numeric application ID.

    Users routinely paste the application ID from the Developer Portal's General Information page
    instead of the bot token (Bot page). A real bot token is dot-separated base64 and never purely
    numeric, so this is a safe, narrow shape check.
    """
    if token and token.strip().isdigit():
        return ("That looks like a numeric application ID, not a bot token. "
                "Paste the bot token from the Discord Developer Portal (Bot page), "
                "not the application ID (General Information page).")
    return None


def _prompt_discord_bot_token(prompt) -> str:
    """Prompt for the bot token, re-prompting once when the answer is a numeric app ID."""
    from hermes_cli.cli_output import print_error
    token = ""
    for _attempt in range(2):
        token = prompt("Discord bot token", password=True)
        if not token:
            return ""
        error = _discord_token_shape_error(token)
        if error is None:
            return token
        print_error(error)
    # Second consecutive numeric answer: trust the user, keep the value.
    return token


def _prompt_checked_token(prompt) -> tuple[str, Optional[DiscordBotCheck]]:
    """A token Discord accepted (with its check), an unverifiable one (offline: ``None`` check),
    or ``("", None)`` when the user gave up."""
    from hermes_cli.cli_output import print_error, print_success, print_warning
    from tools.discord_tool import DiscordAPIError
    for _attempt in range(3):
        token = _prompt_discord_bot_token(prompt).strip()
        if not token:
            return "", None
        try:
            check = check_bot_token(token)
        except DiscordAPIError as exc:
            if exc.status == 401:
                print_error("Discord rejected that token. On the Bot page click Reset Token, "
                            "copy the new token and paste it here.")
                continue
            print_warning(f"Couldn't verify the token (Discord answered {exc.status}); saving it anyway.")
            return token, None
        except OSError as exc:
            print_warning(f"Couldn't reach Discord to verify the token ({exc}); saving it anyway.")
            return token, None
        print_success(f"Token works: this is the bot \"{check.bot_name}\"")
        return token, check
    print_error("Discord rejected three tokens in a row; nothing was saved.")
    return "", None


def _ensure_message_content_intent(token: str, check: DiscordBotCheck, prompt) -> DiscordBotCheck:
    from hermes_cli.cli_output import print_info, print_success, print_warning
    from tools.discord_tool import DiscordAPIError
    for _attempt in range(5):
        if check.message_content:
            break
        print_warning("Message Content Intent is OFF: the bot would receive every message with the text blank.")
        print_info(f"   Turn it on here: {check.bot_settings_url}")
        print_info("   (Privileged Gateway Intents → Message Content Intent → Save Changes)")
        if prompt("Press Enter once it's saved to re-check, or type 'skip'").strip().lower() == "skip":
            return check
        try:
            check = check_bot_token(token)
        except (DiscordAPIError, OSError):
            return check
    if check.message_content:
        print_success("Message Content Intent is on")
    if not check.server_members:
        print_info("Optional: turn on Server Members Intent on the same page if you want to "
                   "allow people by username or role instead of user ID.")
    return check


def _print_invite(check: DiscordBotCheck) -> None:
    from hermes_cli.cli_output import print_info
    print()
    if check.server_count == 0:
        print_info("📨 The bot isn't in any server yet. Open this link to add it to yours:")
    else:
        print_info("📨 Invite link (adds the bot to a server with the permissions Hermes uses):")
    print_info(f"   {check.invite_url}")
    print_info("   Once you share a server with the bot you can also DM it directly.")


def _prompt_allowlist(check: Optional[DiscordBotCheck], prompt, prompt_yes_no) -> None:
    from hermes_cli.cli_output import print_info, print_success
    from hermes_cli.config import save_env_value
    print()
    print_info("🔒 Security: only allowlisted Discord users can talk to your bot.")
    allowed: list = []
    if check and check.owners:
        who = ", ".join(f"@{name}" for _uid, name in check.owners)
        question = (f"Allow yourself ({who}) to talk to the bot?" if len(check.owners) == 1
                    else f"Allow your team ({who}) to talk to the bot?")
        if prompt_yes_no(question, True):
            allowed = [uid for uid, _name in check.owners]
    print_info("   Others need their Discord user ID: Settings → Advanced → Developer Mode,")
    print_info("   then right-click their name → Copy User ID. Usernames work too.")
    raw = prompt("Other allowed users (comma-separated, Enter to skip)" if allowed
                 else "Allowed user IDs or usernames (comma-separated, leave empty for open access)")
    extra = _clean_discord_user_ids(raw or "")
    allowed += [uid for uid in extra if uid not in allowed]
    if check and not check.server_members and any(not uid.isdigit() for uid in extra):
        print_info("   Usernames are resolved when the gateway connects, which needs Server Members Intent: "
                   f"{check.bot_settings_url}")
    if allowed:
        save_env_value("DISCORD_ALLOWED_USERS", ",".join(allowed))
        print_success("Discord allowlist configured")
    else:
        print_info(
            "⚠️  No allowlist set. Discord will deny messages until you set "
            "DISCORD_ALLOWED_USERS, DISCORD_ALLOWED_ROLES, DISCORD_ALLOWED_CHANNELS, "
            "or DISCORD_ALLOW_ALL_USERS=true for open access."
        )


def _check_saved_token(token: str) -> Optional[DiscordBotCheck]:
    from tools.discord_tool import DiscordAPIError
    try:
        return check_bot_token(token) if token else None
    except (DiscordAPIError, OSError):
        return None


def interactive_setup() -> None:
    """Guide the user through Discord bot setup: token (checked live), intents, invite link,
    allowlist (defaults to the bot's owner) and home channel. CLI imports are lazy."""
    from hermes_cli.cli_output import print_header, print_info, print_success, prompt, prompt_yes_no
    from hermes_cli.config import get_env_value, remove_env_value, save_env_value
    from hermes_cli.setup_platforms import declines_reconfigure

    print_header("Discord")
    if declines_reconfigure("Discord", "Reconfigure Discord?", "DISCORD_BOT_TOKEN"):
        if not get_env_value("DISCORD_ALLOWED_USERS"):
            print_info(
                "⚠️  Discord has no user allowlist. With the fail-closed default, "
                "messages are denied unless you configure allowed users, roles, "
                "or channels, or set DISCORD_ALLOW_ALL_USERS=true."
            )
            if prompt_yes_no("Add allowed users now?", True):
                saved = _check_saved_token(get_env_value("DISCORD_BOT_TOKEN") or "")
                _prompt_allowlist(saved, prompt, prompt_yes_no)
        return
    for line in (
        "1. Open https://discord.com/developers/applications → New Application",
        "2. Open the Bot page → Reset Token → copy the token",
        "Hermes checks the token, the intents and the invite link for you next.",
        "Guide: https://hermes-agent.nousresearch.com/docs/user-guide/messaging/discord",
    ):
        print_info(line)
    token, check = _prompt_checked_token(prompt)
    if not token:
        return
    save_env_value("DISCORD_BOT_TOKEN", token)
    print_success("Discord token saved")
    if check is None:
        print_info("Make sure Message Content Intent is on (Bot page → Privileged Gateway Intents), "
                   "or the bot will see blank messages.")
    else:
        check = _ensure_message_content_intent(token, check, prompt)
        _print_invite(check)
    _prompt_allowlist(check, prompt, prompt_yes_no)
    print()
    for line in (
        "📬 Home Channel: where Hermes delivers cron job results,",
        "   cross-platform messages, and notifications.",
        "   Easiest: type /set-home in a Discord channel once the bot is running.",
        "   Or paste a channel ID (Developer Mode → right-click a channel → Copy Channel ID).",
    ):
        print_info(line)
    home_channel = prompt("Home channel ID (leave empty to set later with /set-home)").strip()
    if home_channel:
        save_env_value("DISCORD_HOME_CHANNEL", home_channel)
    elif remove_env_value("DISCORD_HOME_CHANNEL"):
        print_info("Home channel cleared.")

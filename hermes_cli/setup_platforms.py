"""Messaging-platform setup wizards (Telegram, BlueBubbles, webhooks) and the ``hermes setup
gateway`` flow. setup.py re-exports the public names, and tests monkeypatch prompt/print/env
helpers on hermes_cli.setup, so those are imported lazily per function."""

import contextlib
import logging
import re
from pathlib import Path

logger = logging.getLogger("hermes_cli.setup")

_TELEGRAM_BOT_TOKEN_RE = re.compile(r"^\d+:[A-Za-z0-9_-]{30,}$")
_RULE = "━" * 50


def _is_valid_telegram_bot_token(token: str) -> bool:
    return bool(_TELEGRAM_BOT_TOKEN_RE.match(token))


def _profile_name_from_hermes_home(hermes_home) -> str | None:
    """Return the active profile name when HERMES_HOME is a profile dir."""
    return hermes_home.name if hermes_home.parent.name == "profiles" else None


def _setup_telegram_auto_result():
    """Attempt automatic Telegram bot creation via managed QR onboarding."""
    from hermes_cli.setup import get_hermes_home
    try:
        from hermes_cli.telegram_managed_bot import auto_setup_telegram_bot_result
    except ImportError:
        return None
    profile_name: str | None = None
    with contextlib.suppress(Exception):
        profile_name = _profile_name_from_hermes_home(Path(get_hermes_home()))
    return auto_setup_telegram_bot_result(profile_name=profile_name)


def _declines_reconfigure(env_var: str, label: str, question: str) -> bool:
    """True when ``env_var`` is already set and the user does NOT want to reconfigure."""
    from hermes_cli.setup import get_env_value, print_info, prompt_yes_no
    if not get_env_value(env_var):
        return False
    print_info(f"{label}: already configured")
    return not prompt_yes_no(question, False)


def _save_prompted(env_var: str, question: str, *, password: bool = False, success_msg: str | None = None,
                   skip_msg: str | None = None, transform=None) -> str:
    """Prompt, persist the (optionally transformed) answer when non-empty, and report either way.

    ``success_msg`` may reference ``{value}``. Returns the raw answer ("" when skipped).
    """
    from hermes_cli.setup import print_success, print_warning, prompt, save_env_value
    value = prompt(question, password=password)
    if value:
        save_env_value(env_var, transform(value) if transform else value)
        if success_msg:
            print_success(success_msg.format(value=value))
    elif skip_msg:
        print_warning(skip_msg)
    return value


def _save_allowlist(env_var: str, users: str, success_msg: str) -> None:
    """Strip spaces, persist the allowlist, and confirm."""
    from hermes_cli.setup import print_success, save_env_value
    save_env_value(env_var, users.replace(" ", ""))
    print_success(success_msg)


def _prompt_allowlist(env_var: str, question: str, success_msg: str, open_msg: str, preset: str | None = None) -> str:
    """Persist ``preset`` (or the prompted answer) as an allowlist, warning when it stays open."""
    from hermes_cli.setup import print_info, prompt
    users = prompt(question) if preset is None else preset
    if users:
        _save_allowlist(env_var, users, success_msg)
    else:
        print_info(open_msg)
    return users.replace(" ", "")


def _save_port(env_var: str, value: str, default: str) -> None:
    """Persist ``value`` as an int port; warn (keeping ``default``) when it isn't one."""
    from hermes_cli.setup import print_success, print_warning, save_env_value
    if not value:
        return
    try:
        save_env_value(env_var, str(int(value)))
        print_success(f"Webhook port set to {value}")
    except ValueError:
        print_warning(f"Invalid port number, using default {default}")


def _prompt_telegram_bot_token() -> str | None:
    from hermes_cli.setup import print_error, print_info, prompt
    print_info("Create a bot via @BotFather on Telegram")
    while True:
        token = prompt("Telegram bot token", password=True)
        if not token or _is_valid_telegram_bot_token(token):
            return token or None
        print_error("Invalid token format. Expected: <numeric_id>:<alphanumeric_hash> "
                    "(e.g., 123456789:ABCdefGHI-jklMNOpqrSTUvwxYZ)")


def _telegram_allowlist_nudge() -> None:
    """Existing config kept as-is: warn when it has no user allowlist."""
    from hermes_cli.setup import get_env_value, print_info, prompt, prompt_yes_no
    if get_env_value("TELEGRAM_ALLOWED_USERS"):
        return
    print_info("⚠️  Telegram has no user allowlist - anyone can use your bot!")
    if prompt_yes_no("Add allowed users now?", True):
        print_info("   To find your Telegram user ID: message @userinfobot")
        allowed_users = prompt("Allowed user IDs (comma-separated)")
        if allowed_users:
            _save_allowlist("TELEGRAM_ALLOWED_USERS", allowed_users, "Telegram allowlist configured")


def _obtain_telegram_token():
    """Return (token, setup_result); auto flow first when chosen, else manual paste."""
    from hermes_cli.setup import _info, print_error, prompt
    _info("How would you like to create your Telegram bot?", None,
          "  [1] Automatic (recommended)",
          "      Scan a QR code → confirm in Telegram → done.",
          "      No token copy-paste needed.", None,
          "  [2] Manual",
          "      Create a bot via @BotFather yourself and paste the token.", None)
    token = setup_result = None
    if prompt("Choice [1/2]", default="1").strip() == "1":
        setup_result = _setup_telegram_auto_result()
        if setup_result:
            token = setup_result.token
            if not _is_valid_telegram_bot_token(token):
                print_error("Automatic setup returned an invalid Telegram bot token.")
                token = setup_result = None
        if not token:
            _info(None, "Falling back to manual setup...", None)
    if not token:
        token = _prompt_telegram_bot_token()
    return token, setup_result


def _setup_telegram():
    """Configure Telegram bot credentials and allowlist."""
    from hermes_cli.setup import _info, print_info, print_header, print_success, prompt, prompt_yes_no, save_env_value
    print_header("Telegram")
    if _declines_reconfigure("TELEGRAM_BOT_TOKEN", "Telegram", "Reconfigure Telegram?"):
        _telegram_allowlist_nudge()
        return
    token, setup_result = _obtain_telegram_token()
    if not token:
        return
    save_env_value("TELEGRAM_BOT_TOKEN", token)
    print_success("Telegram token saved")
    _info(None, "🔒 Security: Restrict who can use your bot",
          "   To find your Telegram user ID:",
          "   1. Message @userinfobot on Telegram",
          "   2. It will reply with your numeric ID (e.g., 123456789)", None)
    allowed_users = None
    detected_id = str(getattr(setup_result, "owner_user_id", None) or "")
    if detected_id:
        print_success(f"Detected your Telegram user ID: {detected_id}")
        if prompt_yes_no("Allow this Telegram account to use the bot?", True):
            extra = prompt("Additional allowed user IDs (comma-separated, optional)")
            allowed_users = ",".join(dict.fromkeys([detected_id, *filter(None, extra.replace(" ", "").split(","))]))
    allowed_users = _prompt_allowlist(
        "TELEGRAM_ALLOWED_USERS", "Allowed user IDs (comma-separated, leave empty for open access)",
        "Telegram allowlist configured - only listed users can use the bot",
        "⚠️  No allowlist set - anyone who finds your bot can use it!", preset=allowed_users)
    _info(None, "📬 Home Channel: where Hermes delivers cron job results,",
          "   cross-platform messages, and notifications.",
          "   For Telegram DMs, this is your user ID (same as above).")
    first_user_id = allowed_users.split(",")[0].strip() if allowed_users else ""
    if not first_user_id:
        print_info("   You can also set this later by typing /set-home in your Telegram chat.")
        _save_prompted("TELEGRAM_HOME_CHANNEL", "Home channel ID (leave empty to set later)")
    elif prompt_yes_no(f"Use your user ID ({first_user_id}) as the home channel?", True):
        save_env_value("TELEGRAM_HOME_CHANNEL", first_user_id)
        print_success(f"Telegram home channel set to {first_user_id}")
    else:
        _save_prompted("TELEGRAM_HOME_CHANNEL", "Home channel ID (or leave empty to set later with /set-home in Telegram)")


# _setup_slack and _write_slack_manifest_and_instruct moved to the slack plugin:
# plugins/platforms/slack/adapter.py::interactive_setup (registered via setup_fn and dispatched through the
# plugin path). #41112 / #3823.
def _setup_bluebubbles():
    """Configure BlueBubbles iMessage gateway."""
    from hermes_cli.setup import _info, print_header, print_success, prompt, prompt_yes_no
    print_header("BlueBubbles (iMessage)")
    if _declines_reconfigure("BLUEBUBBLES_SERVER_URL", "BlueBubbles", "Reconfigure BlueBubbles?"):
        return
    _info("Connects Hermes to iMessage via BlueBubbles — a free, open-source",
          "macOS server that bridges iMessage to any device.",
          "   Requires a Mac running BlueBubbles Server v1.0.0+",
          "   Download: https://bluebubbles.app/", None,
          "In BlueBubbles Server → Settings → API, note your Server URL and Password.", None)
    for label, env_var, secret, what, transform in (
        ("BlueBubbles server URL (e.g. http://192.168.1.10:1234)", "BLUEBUBBLES_SERVER_URL", False, "Server URL",
         lambda v: v.rstrip("/")),
        ("BlueBubbles server password", "BLUEBUBBLES_PASSWORD", True, "Password", None),
    ):
        if not _save_prompted(env_var, label, password=secret, transform=transform,
                              skip_msg=f"{what} is required — skipping BlueBubbles setup"):
            return
    print_success("BlueBubbles credentials saved")
    _info(None, "🔒 Security: Restrict who can message your bot",
          "   Use iMessage addresses: email (user@icloud.com) or phone (+15551234567)", None)
    _prompt_allowlist("BLUEBUBBLES_ALLOWED_USERS", "Allowed iMessage addresses (comma-separated, leave empty for open access)",
                      "BlueBubbles allowlist configured", "⚠️  No allowlist set — anyone who can iMessage you can use the bot!")
    _info(None, "📬 Home Channel: phone or email for cron job delivery and notifications.",
          "   You can also set this later with /set-home in your iMessage chat.")
    _save_prompted("BLUEBUBBLES_HOME_CHANNEL", "Home channel address (leave empty to set later)")
    _info(None, "Advanced settings (defaults are fine for most setups):")
    if prompt_yes_no("Configure webhook listener settings?", False):
        _save_port("BLUEBUBBLES_WEBHOOK_PORT", prompt("Webhook listener port (default: 8645)"), "8645")
    _info(None, "Requires the BlueBubbles Private API helper for typing indicators,",
          "read receipts, and tapback reactions. Basic messaging works without it.",
          "   Install: https://docs.bluebubbles.app/helper-bundle/installation")


def _setup_webhooks():
    """Configure webhook integration."""
    from hermes_cli.setup import _info, print_header, print_success, print_warning, prompt, save_env_value
    print_header("Webhooks")
    if _declines_reconfigure("WEBHOOK_ENABLED", "Webhooks", "Reconfigure webhooks?"):
        return
    print()
    print_warning("⚠  Webhook and SMS platforms require exposing gateway ports to the")
    print_warning("   internet. For security, run the gateway in a sandboxed environment")
    print_warning("   (Docker, VM, etc.) to limit blast radius from prompt injection.")
    print()
    _info("   Full guide: https://hermes-agent.nousresearch.com/docs/user-guide/messaging/webhooks/", None)
    _save_port("WEBHOOK_PORT", prompt("Webhook port (default 8644)"), "8644")
    _save_prompted("WEBHOOK_SECRET", "Global HMAC secret (shared across all routes)", password=True,
                   success_msg="Webhook secret saved",
                   skip_msg="No secret set — you must configure per-route secrets in config.yaml")
    save_env_value("WEBHOOK_ENABLED", "true")
    print()
    print_success("Webhooks enabled! Next steps:")
    from hermes_constants import display_hermes_home as _dhh
    _info(f"   1. Define webhook routes in {_dhh()}/config.yaml",
          "   2. Point your service (GitHub, GitLab, etc.) at:",
          "      http://your-server:8644/webhooks/<route-name>", None,
          "   Route configuration guide:",
          "   https://hermes-agent.nousresearch.com/docs/user-guide/messaging/webhooks/#configuring-routes",
          None,
          # Printed twice upstream; kept verbatim for output parity.
          "   Open config in your editor:  hermes config edit",
          "   Open config in your editor:  hermes config edit")


# (platform label, credential env var, home-channel env vars — any one satisfies)
_HOME_CHANNEL_CHECKS = (
    ("Telegram", "TELEGRAM_BOT_TOKEN", ("TELEGRAM_HOME_CHANNEL",)), ("Discord", "DISCORD_BOT_TOKEN", ("DISCORD_HOME_CHANNEL",)),
    ("Slack", "SLACK_BOT_TOKEN", ("SLACK_HOME_CHANNEL",)), ("BlueBubbles", "BLUEBUBBLES_SERVER_URL", ("BLUEBUBBLES_HOME_CHANNEL",)),
    ("QQBot", "QQ_APP_ID", ("QQBOT_HOME_CHANNEL", "QQ_HOME_CHANNEL")),
)


def _is_progress(status: str) -> bool:
    """A platform counts as configured unless its status says otherwise."""
    s = status.lower()
    return not (s == "not configured" or s.startswith(("partially", "plugin disabled")))


def _warn_missing_home_channels() -> None:
    """Platforms with a token but no home channel."""
    from hermes_cli.setup import get_env_value, _info, print_warning
    missing_home = [
        plat for plat, token_var, home_vars in _HOME_CHANNEL_CHECKS
        if get_env_value(token_var) and not any(get_env_value(v) for v in home_vars)]
    if not missing_home:
        return
    print()
    print_warning(f"No home channel set for: {', '.join(missing_home)}")
    _info("   Without a home channel, cron jobs and cross-platform",
          "   messages can't be delivered to those platforms.",
          "   Set one later with /set-home in your chat, or:",
          *(f"     hermes config set {plat.upper()}_HOME_CHANNEL <channel_id>" for plat in missing_home))


def _restart_running_gateway(any_messaging: bool, supports_systemd: bool) -> None:
    """Already running: offer a restart only when this pass may have changed platform config —
    a restart interrupts any active session, so it stays behind a prompt."""
    from hermes_cli.setup import print_error, prompt_yes_no
    from hermes_cli.gateway import (
        systemd_restart, launchd_restart, UserSystemdUnavailableError, SystemScopeRequiresRootError,
        _system_scope_wizard_would_need_root, _print_system_scope_remediation,
    )
    import platform as _platform
    if supports_systemd and _system_scope_wizard_would_need_root():
        _print_system_scope_remediation("restart")
        return
    if not (any_messaging and prompt_yes_no("  Restart the gateway to pick up changes?", True)):
        return
    try:
        if supports_systemd:
            systemd_restart()
        elif _platform.system() == "Darwin":
            launchd_restart()
        elif _platform.system() == "Windows":
            from hermes_cli import gateway_windows
            gateway_windows.restart()
    except UserSystemdUnavailableError as e:
        print_error("  Restart failed — user systemd not reachable:")
        for line in str(e).splitlines():
            print(f"  {line}")
    except SystemScopeRequiresRootError as e:
        # Defense in depth: a race (unit file appearing mid-run) can slip past the pre-check;
        # this used to sys.exit(1) the whole wizard.
        print_error(f"  Restart failed: {e}")
        _print_system_scope_remediation("restart")
    except Exception as e:
        print_error(f"  Restart failed: {e}")


def setup_gateway(config: dict):
    """Configure messaging platform integrations."""
    from hermes_cli.setup import _info, print_header, print_info, print_success, prompt_checklist
    from hermes_cli.gateway import _all_platforms, _platform_status, _configure_platform
    print_header("Messaging Platforms")
    _info("Connect to messaging platforms to chat with Hermes from anywhere.",
          "Toggle with Space, confirm with Enter.", None)
    platforms = _all_platforms()

    # Build checklist, pre-selecting already-configured platforms.
    statuses = [_platform_status(plat) for plat in platforms]
    items = [f"{plat['emoji']} {plat['label']}  ({status})" for plat, status in zip(platforms, statuses)]
    pre_selected = [i for i, status in enumerate(statuses) if status == "configured"]
    selected = prompt_checklist("Select platforms to configure:", items, pre_selected)
    if not selected:
        print_info("No platforms selected. Run 'hermes setup gateway' later to configure.")
    for idx in selected or ():
        _configure_platform(platforms[idx])

    # Any platform (built-in or plugin) configured in this pass — via ``_platform_status`` so
    # plugin platforms like IRC are counted without another hard-coded env-var list.
    any_messaging = any(_is_progress(_platform_status(p)) for p in _all_platforms())
    if any_messaging:
        print()
        print_info(_RULE)
        print_success("Messaging platforms configured!")
        _warn_missing_home_channels()

    # Gateway service setup runs UNCONDITIONALLY — a gateway with zero platforms is a supported
    # mode (cron keeps running; adapters come up once tokens are added via `hermes import` /
    # `hermes setup gateway`). Gating it on messaging config left install-then-import machines
    # with cron jobs and bot tokens but no process to serve them.
    from hermes_cli.gateway import _is_service_running, supports_systemd_services, ensure_gateway_service
    supports_systemd = supports_systemd_services()
    print()
    if _is_service_running():
        _restart_running_gateway(any_messaging, supports_systemd)
    else:
        # Not running: install (if needed) and start, no questions asked.
        ensure_gateway_service(context="setup")
    print_info(_RULE)

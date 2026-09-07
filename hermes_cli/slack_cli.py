"""``hermes slack manifest`` — generate the Slack app manifest JSON that registers every gateway
command as a native Slack slash (``/btw``, ``/stop``, ``/model``, …)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SLACK_LONG_DESCRIPTION_MIN_CHARACTERS = 175
SLACK_LONG_DESCRIPTION_MAX_CHARACTERS = 4000


def _build_full_manifest(
    bot_name: str, bot_description: str, include_assistant: bool = True,
    messaging_experience: str | None = None, long_description: str | None = None) -> dict:
    """Build a full Slack manifest: display info + slash list from ``COMMAND_REGISTRY``.

    Other sections (OAuth scopes, socket mode) are sensible Hermes defaults, tweakable in the Slack
    UI after pasting.
    """
    from hermes_cli.commands_platforms import slack_app_manifest
    if messaging_experience is None:
        messaging_experience = "assistant" if include_assistant else "none"
    messaging_experience = str(messaging_experience).strip().lower()
    if messaging_experience not in {"assistant", "agent", "none"}:
        raise ValueError("messaging_experience must be one of: assistant, agent, none")

    features = {
        "app_home": {
            "home_tab_enabled": False,
            "messages_tab_enabled": True,
            "messages_tab_read_only_enabled": False},
        "bot_user": {"display_name": bot_name[:80], "always_online": True},
        "slash_commands": slack_app_manifest()["features"]["slash_commands"]}

    bot_scopes = [
        "app_mentions:read", "channels:history", "channels:read", "chat:write", "commands",
        "files:read", "files:write", "groups:history", "groups:read", "im:history", "im:read",
        "im:write", "mpim:history", "mpim:read", "reactions:read", "users:read"]
    bot_events = [
        "app_mention", "message.channels", "message.groups", "message.im", "message.mpim",
        "reaction_added", "reaction_removed"]

    if messaging_experience == "assistant":
        features["assistant_view"] = {
            "assistant_description": "Chat with Hermes in threads and DMs."}
        bot_scopes.append("assistant:write")
        bot_events.extend(["assistant_thread_context_changed", "assistant_thread_started"])
    elif messaging_experience == "agent":
        features["agent_view"] = {"agent_description": "Chat with Hermes in Slack Messages."}
        bot_scopes.append("assistant:write")
        # Slack includes current viewing context in Agent DM events only after this subscription
        # is enabled; the adapter uses it to preserve the referred channel across the agent turn.
        bot_events.extend(["app_context_changed", "app_home_opened"])

    bot_scopes.sort()
    bot_events.sort()

    display_information = {
        "name": bot_name[:35],
        "description": (bot_description or "Your Hermes agent on Slack")[:140],
        "background_color": "#1a1a2e"}
    if long_description is not None:
        display_information["long_description"] = long_description

    return {
        "_metadata": {"major_version": 1, "minor_version": 1},
        "display_information": display_information,
        "features": features,
        "oauth_config": {"scopes": {"bot": bot_scopes}},
        "settings": {
            "event_subscriptions": {"bot_events": bot_events},
            "interactivity": {"is_enabled": True},
            "org_deploy_enabled": False,
            "socket_mode_enabled": True,
            "token_rotation_enabled": False}}


def slack_manifest_command(args) -> int:
    """Print or write a Slack app manifest JSON (flags documented in ``hermes_cli/main.py``)."""
    name = getattr(args, "name", None) or "Hermes"
    description = getattr(args, "description", None) or "Your Hermes agent on Slack"
    long_description = getattr(args, "long_description", None)
    long_description_file = getattr(args, "long_description_file", None)
    slashes_only = getattr(args, "slashes_only", False)

    def fail(msg: str) -> int:
        print(f"hermes slack manifest: {msg}", file=sys.stderr)
        return 2

    if slashes_only and (long_description is not None or long_description_file is not None):
        return fail("long description options cannot be used with --slashes-only")
    if long_description_file is not None:
        source_arg = str(long_description_file)
        try:
            with Path(source_arg).expanduser().open("r", encoding="utf-8", newline="") as handle:
                long_description = handle.read()
        except (OSError, UnicodeError, RuntimeError) as exc:
            return fail(f"cannot read long description from {source_arg}: {exc}")
    if long_description is not None:
        n = len(long_description)
        if n < SLACK_LONG_DESCRIPTION_MIN_CHARACTERS:
            return fail(f"long description must be at least "
                        f"{SLACK_LONG_DESCRIPTION_MIN_CHARACTERS} characters (got {n})")
        if n > SLACK_LONG_DESCRIPTION_MAX_CHARACTERS:
            return fail(f"long description must be at most "
                        f"{SLACK_LONG_DESCRIPTION_MAX_CHARACTERS} characters (got {n})")
    if getattr(args, "agent_view", False):
        messaging_experience = "agent"
    elif getattr(args, "no_assistant", False):
        messaging_experience = "none"
    else:
        messaging_experience = "assistant"

    if slashes_only:
        from hermes_cli.commands_platforms import slack_app_manifest
        manifest = slack_app_manifest()["features"]["slash_commands"]
    else:
        manifest = _build_full_manifest(
            name, description, messaging_experience=messaging_experience,
            long_description=long_description)
    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"

    write_target = getattr(args, "write", None)
    if write_target is None:
        sys.stdout.write(payload)
        return 0
    if isinstance(write_target, bool) and write_target:  # bare --write → default location
        from hermes_constants import get_hermes_home
        target = Path(get_hermes_home()) / "slack-manifest.json"
    else:
        target = Path(write_target).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    print(f"Slack manifest written to: {target}", file=sys.stderr)
    print(
        "\nNext steps:\n"
        "  1. Open https://api.slack.com/apps and pick your Hermes app\n"
        "     (or create a new one: Create New App → From an app manifest).\n"
        f"  2. Features → App Manifest → paste the contents of\n"
        f"     {target}\n"
        "  3. Save; Slack will prompt to reinstall the app if scopes or\n"
        "     slash commands changed.\n"
        "  4. Make sure Socket Mode is enabled and you have a bot token\n"
        "     (xoxb-...) and app token (xapp-...) configured via\n"
        "     `hermes setup`.\n", file=sys.stderr)
    return 0


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----

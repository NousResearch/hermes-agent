"""``hermes slack ...`` CLI subcommands.

Today only ``hermes slack manifest`` is implemented — it generates the
Slack app manifest JSON for registering every gateway command as a native
Slack slash (``/btw``, ``/stop``, ``/model``, …) so users get the same
first-class slash UX Discord and Telegram already have.

Typical workflow::

    $ hermes slack manifest > slack-manifest.json
    # or:
    $ hermes slack manifest --write

Then paste the printed JSON into the Slack app config (Features → App
Manifest → Edit) and click Save. Slack diffs the manifest and prompts
for reinstall when scopes/commands change.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any


def _slack_ingress_settings(config: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize bounded ingress policy settings from ``config.yaml``."""
    root = config if isinstance(config, dict) else {}
    ingress: dict[str, Any] = {}
    slack = root.get("slack")
    if isinstance(slack, dict) and isinstance(slack.get("ingress"), dict):
        ingress = slack["ingress"]
    else:
        platforms = root.get("platforms")
        platform_slack = platforms.get("slack") if isinstance(platforms, dict) else None
        extra = platform_slack.get("extra") if isinstance(platform_slack, dict) else None
        if isinstance(extra, dict) and isinstance(extra.get("ingress"), dict):
            ingress = extra["ingress"]

    try:
        ttl_days = float(ingress.get("follow_ttl_days", 30))
        max_threads = int(ingress.get("max_followed_threads", 10_000))
    except (TypeError, ValueError) as exc:
        raise ValueError("Slack ingress TTL and thread cap must be numeric") from exc
    if ttl_days <= 0 or max_threads <= 0:
        raise ValueError("Slack ingress TTL and thread cap must be positive")

    def _tokens(value: Any, *, strip_colons: bool = False) -> set[str]:
        if isinstance(value, str):
            values = value.split(",")
        elif isinstance(value, (list, tuple, set)):
            values = value
        else:
            values = ()
        result: set[str] = set()
        for item in values:
            token = str(item).strip()
            if strip_colons:
                token = token.strip(":").lower()
            if token:
                result.add(token)
        return result

    return {
        "ttl_seconds": ttl_days * 24 * 60 * 60,
        "max_threads": max_threads,
        "reaction_user_ids": _tokens(ingress.get("reaction_user_ids")),
        "reaction_names": _tokens(
            ingress.get("reaction_names"), strip_colons=True
        ),
    }


def slack_ingress_command(args) -> int:
    """Run the machine-singleton Slack ingress/Relay sidecar."""
    from gateway.config import Platform, load_gateway_config
    from hermes_cli.config import load_config
    from hermes_constants import get_default_hermes_root, get_hermes_home
    from plugins.platforms.slack.adapter import SlackAdapter
    from plugins.platforms.slack.ingress import (
        FollowStore,
        SlackIngressPolicy,
        SlackIngressServer,
    )

    hermes_home = get_hermes_home()
    try:
        settings = _slack_ingress_settings(load_config())
        gateway_config = load_gateway_config()
        slack_config = gateway_config.platforms.get(Platform.SLACK)
        if slack_config is not None and slack_config.enabled:
            raise RuntimeError(
                "set slack.enabled: false before starting the ingress sidecar; "
                "the Gateway and sidecar must not both own Slack Socket Mode"
            )
        if slack_config is None or not slack_config.token:
            raise RuntimeError(
                "SLACK_BOT_TOKEN is required for `hermes slack ingress`"
            )
        state_path = Path(
            getattr(args, "state", None) or hermes_home / "slack-ingress.db"
        ).expanduser()
        policy = SlackIngressPolicy(
            FollowStore(
                state_path,
                ttl_seconds=settings["ttl_seconds"],
                max_threads=settings["max_threads"],
            ),
            reaction_user_ids=settings["reaction_user_ids"],
            reaction_names=settings["reaction_names"],
        )
        server = SlackIngressServer(
            SlackAdapter(slack_config),
            policy,
            host=str(getattr(args, "host", "127.0.0.1")),
            port=int(getattr(args, "port", 8791)),
            lock_path=get_default_hermes_root() / "slack-ingress.lock",
        )

        async def _serve() -> None:
            await server.start()
            print(
                f"Slack ingress listening at {server.url}/relay; "
                "Slack connects only after the Gateway Relay handshake.",
                file=sys.stderr,
            )
            try:
                await server.wait_closed()
            finally:
                await server.stop()

        asyncio.run(_serve())
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Slack ingress failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _build_full_manifest(
    bot_name: str,
    bot_description: str,
    include_assistant: bool = True,
    messaging_experience: str | None = None,
) -> dict:
    """Build a full Slack manifest merging display info + our slash list.

    The slash-command list is always generated from ``COMMAND_REGISTRY`` so
    it stays in sync with the rest of Hermes. Other manifest sections
    (display info, OAuth scopes, socket mode) are set to sensible defaults
    for a Hermes deployment — users can tweak them in the Slack UI after
    pasting.

    By default, this keeps Hermes on Slack's older Assistant messaging
    experience (``assistant_view``) for backward compatibility. Pass
    ``messaging_experience="agent"`` (``--agent-view``) to emit Slack's Agent
    messaging experience (``agent_view`` + ``app_home_opened``). Pass
    ``include_assistant=False`` or ``messaging_experience="none"``
    (``--no-assistant``) to omit Slack AI messaging features and get a flat DM
    surface where ``/help``, ``/new``, etc. work inline.
    """
    from hermes_cli.commands import slack_app_manifest

    if messaging_experience is None:
        messaging_experience = "assistant" if include_assistant else "none"
    messaging_experience = str(messaging_experience).strip().lower()
    if messaging_experience not in {"assistant", "agent", "none"}:
        raise ValueError(
            "messaging_experience must be one of: assistant, agent, none"
        )

    partial = slack_app_manifest()
    slashes = partial["features"]["slash_commands"]

    features = {
        "app_home": {
            "home_tab_enabled": False,
            "messages_tab_enabled": True,
            "messages_tab_read_only_enabled": False,
        },
        "bot_user": {
            "display_name": bot_name[:80],
            "always_online": True,
        },
        "slash_commands": slashes,
    }

    bot_scopes = [
        "app_mentions:read",
        "channels:history",
        "channels:read",
        "chat:write",
        "commands",
        "files:read",
        "files:write",
        "groups:history",
        "groups:read",
        "im:history",
        "im:read",
        "im:write",
        "mpim:history",
        "mpim:read",
        "reactions:read",
        "users:read",
    ]

    bot_events = [
        "app_mention",
        "message.channels",
        "message.groups",
        "message.im",
        "message.mpim",
        "reaction_added",
    ]

    if messaging_experience == "assistant":
        features["assistant_view"] = {
            "assistant_description": "Chat with Hermes in threads and DMs.",
        }
        bot_scopes.append("assistant:write")
        bot_events.extend(
            [
                "assistant_thread_context_changed",
                "assistant_thread_started",
            ]
        )
    elif messaging_experience == "agent":
        features["agent_view"] = {
            "agent_description": "Chat with Hermes in Slack Messages.",
        }
        bot_scopes.append("assistant:write")
        # Slack includes current viewing context in Agent DM events only after
        # this subscription is enabled; the adapter consumes that context to
        # preserve the referred channel across the agent turn.
        bot_events.extend(["app_context_changed", "app_home_opened"])

    bot_scopes.sort()
    bot_events.sort()

    return {
        "_metadata": {
            "major_version": 1,
            "minor_version": 1,
        },
        "display_information": {
            "name": bot_name[:35],
            "description": (bot_description or "Your Hermes agent on Slack")[:140],
            "background_color": "#1a1a2e",
        },
        "features": features,
        "oauth_config": {
            "scopes": {
                "bot": bot_scopes,
            },
        },
        "settings": {
            "event_subscriptions": {
                "bot_events": bot_events,
            },
            "interactivity": {
                "is_enabled": True,
            },
            "org_deploy_enabled": False,
            "socket_mode_enabled": True,
            "token_rotation_enabled": False,
        },
    }


def slack_manifest_command(args) -> int:
    """Print or write a Slack app manifest JSON.

    Flags (all parsed in ``hermes_cli/main.py``):
      --write [PATH]  Write to file instead of stdout (default path:
                      ``$HERMES_HOME/slack-manifest.json``)
      --name NAME     Override the bot display name (default: "Hermes")
      --description DESC  Override the bot description
      --slashes-only  Emit only the ``features.slash_commands`` array (for
                      merging into an existing manifest manually)
      --no-assistant  Omit Slack AI Assistant mode (assistant_view feature,
                      assistant:write scope, assistant_thread_* events) so
                      DMs render as a flat chat where bare slash commands
                      work inline instead of the Assistant thread pane.
      --agent-view    Use Slack's Agent messaging experience (agent_view,
                      app_home_opened + message.im) instead of the legacy
                      Assistant messaging experience.
    """
    name = getattr(args, "name", None) or "Hermes"
    description = getattr(args, "description", None) or "Your Hermes agent on Slack"
    if getattr(args, "agent_view", False):
        messaging_experience = "agent"
    elif getattr(args, "no_assistant", False):
        messaging_experience = "none"
    else:
        messaging_experience = "assistant"

    if getattr(args, "slashes_only", False):
        from hermes_cli.commands import slack_app_manifest

        manifest = slack_app_manifest()["features"]["slash_commands"]
    else:
        manifest = _build_full_manifest(
            name,
            description,
            messaging_experience=messaging_experience,
        )

    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"

    write_target = getattr(args, "write", None)
    if write_target is not None:
        if isinstance(write_target, bool) and write_target:
            # --write with no value → default location
            try:
                from hermes_constants import get_hermes_home

                target = Path(get_hermes_home()) / "slack-manifest.json"
            except Exception:
                target = Path(os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")) / "slack-manifest.json"
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
            "     `hermes setup`.\n",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(payload)
    return 0

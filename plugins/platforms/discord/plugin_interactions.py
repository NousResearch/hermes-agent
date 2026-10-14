"""Discord native command registration for opt-in typed plugin interactions."""

from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger("plugins.platforms.discord")


def register_plugin_interaction_commands(
    adapter: Any, tree: Any, discord_module: Any, already_registered: set[str], slot_cap: int,
) -> int:
    """Register typed plugin commands without changing legacy slash-command dispatch."""
    from hermes_cli.plugin_interaction_api import get_plugin_discord_interaction_commands
    from hermes_cli.plugin_interactions import (
        PrivateInteractionUnavailable, dispatch_plugin_interaction,
    )

    try:
        entries = get_plugin_discord_interaction_commands()
    except Exception:
        logger.exception("Discord plugin interaction command discovery failed")
        return 0

    added = 0
    for name, entry in entries.items():
        discord_name = name.lower()[:32]
        if discord_name in already_registered or len(already_registered) >= slot_cap:
            continue
        callback = entry.get("handler")
        if not callable(callback):
            logger.warning("Discord interaction command /%s has no callable handler; not registered", name)
            continue
        has_args = bool(str(entry.get("args_hint") or "").strip())

        async def _handler(interaction, args: str = "", *, _name=name, _callback=callback, _has_args=has_args):
            if not await adapter._check_slash_authorization(interaction, f"/{_name}"):
                return
            try:
                await dispatch_plugin_interaction(
                    interaction=interaction,
                    adapter=adapter,
                    command_name=_name,
                    raw_args=args if _has_args else "",
                    handler=_callback,
                )
            except PrivateInteractionUnavailable:
                # The helper never sends public output when the private transport is unavailable.
                return

        parameters = [inspect.Parameter(
            "interaction", inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=discord_module.Interaction,
        )]
        if has_args:
            parameters.append(inspect.Parameter(
                "args", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str, default="",
            ))
        _handler.__name__ = f"plugin_interaction_{discord_name.replace('-', '_')}"
        _handler.__signature__ = inspect.Signature(parameters)
        if has_args:
            _handler = discord_module.app_commands.describe(
                args=str(entry.get("args_hint") or "Arguments"),
            )(_handler)
        command = discord_module.app_commands.Command(
            name=discord_name,
            description=str(entry.get("description") or f"Plugin interaction: {name}")[:100],
            callback=_handler,
        )
        try:
            tree.add_command(command)
        except Exception:
            logger.exception("Discord interaction command /%s could not be registered", name)
            continue
        already_registered.add(discord_name)
        added += 1
    return added

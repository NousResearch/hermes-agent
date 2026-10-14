"""Owner-aware admission for native plugin CLI registrations."""

from __future__ import annotations

from collections.abc import Callable
from logging import Logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.plugins import PluginContext
    from hermes_cli.plugins_ledger import PluginRegistration


def register_cli_command(
    context: PluginContext, name: str, help: str, setup_fn: Callable,
    handler_fn: Callable | None, description: str, *, logger: Logger,
) -> PluginRegistration | None:
    """Preserve the first foreign owner; the caller holds the replacement transaction.

    Same-owner replacements use the existing lease/ledger path so disposal can
    restore a still-active predecessor. A rejection creates no handle or lease.
    """
    existing = context._manager._cli_commands.get(name)
    if existing is not None:
        owner = existing.get("plugin_key", existing.get("plugin"))
        if owner != context.plugin_id:
            logger.warning(
                "Plugin %r CLI command %r skipped: command collision with plugin %r",
                context.plugin_id, name, owner,
            )
            return None
    entry = {
        "name": name, "help": help, "description": description, "setup_fn": setup_fn,
        "handler_fn": handler_fn, "plugin": context.manifest.name, "plugin_key": context.plugin_id,
    }
    return context._register_entry("cli_command", name, context._manager._cli_commands, entry,
                                   "Plugin %s registered CLI command: %s", name)

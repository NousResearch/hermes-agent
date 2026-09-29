"""CLI credential commands handler for the /creds slash command.

This module provides the processing logic for the /creds slash command
in the Hermes CLI. It handles set, list, show, remove, clear, and inject
operations, interacting with the encrypted credential store.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from agent.credential_store import get_store, CredentialStore

logger = logging.getLogger(__name__)


def _redact_value(value: str, chars_to_show: int = 4) -> str:
    """Return a masked version of a value for display."""
    if len(value) <= chars_to_show:
        return "****"
    return value[:chars_to_show] + "****"


def handle_creds_command(args: str, cli=None) -> str:
    """Process a /creds slash command.

    Args:
        args: The full argument string after '/creds'.
        cli: Optional HermesCLI instance (for interactive prompts).

    Returns:
        Human-readable result string to show to the user.
    """
    if not args or not args.strip():
        return _handle_creds_list()

    parts = args.strip().split(None, 1)
    subcommand = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    handlers = {
        "list": _handle_creds_list,
        "show": lambda: _handle_creds_show(rest),
        "set": lambda: _handle_creds_set(rest, cli),
        "remove": lambda: _handle_creds_remove(rest),
        "delete": lambda: _handle_creds_remove(rest),
        "clear": _handle_creds_clear,
        "inject": lambda: _handle_creds_inject(rest),
    }

    handler = handlers.get(subcommand)
    if handler:
        return handler()

    # If subcommand is not recognized, treat it as /creds list filtered
    return f"Unknown subcommand: /creds {subcommand}\n\nUsage:\n  /creds               List stored credentials\n  /creds list          List stored credentials\n  /creds show <name>   Show a credential's value\n  /creds set <name> [value]  Set a credential (prompts if value omitted)\n  /creds remove <name> Remove a credential\n  /creds clear         Remove all credentials\n  /creds inject <name> Make credential available as env var"


def _handle_creds_list() -> str:
    """Handle /creds list - show all credential names and metadata."""
    store = get_store()
    # Register values with redactor even on list so output doesn't leak them
    entries = store.list()
    for entry in entries:
        value = store.get(entry["name"])
        if value:
            try:
                from agent.redact import register_secret
                register_secret(value)
            except Exception:
                pass

    if not entries:
        return "No credentials stored. Use /creds set <name> [value] to store one."

    lines = [f"Stored credentials ({len(entries)}):"]
    lines.append("")
    for e in entries:
        lines.append(f"  {e['name']}")

    lines.append("")
    lines.append("Use /creds show <name> to reveal a value.")
    return "\n".join(lines)


def _handle_creds_show(name: str) -> str:
    """Handle /creds show <name> - display a credential's value."""
    if not name:
        return "Usage: /creds show <name>"

    value = get_store().get(name.strip())
    if value is None:
        return f"Credential '{name}' not found."

    # Register with redactor so output is scrubbed if accidentally echoed
    try:
        from agent.redact import register_secret
        register_secret(value)
    except Exception:
        pass

    return f"{name}: {value}"


def _handle_creds_set(arg: str, cli=None) -> str:
    """Handle /creds set <name> [value] - store a credential."""
    if not arg:
        return "Usage: /creds set <name> [value]\n  If value is omitted, you'll be prompted (masked input)."

    parts = arg.strip().split(None, 1)
    name = parts[0].strip()
    value = parts[1] if len(parts) > 1 else None

    # Validate name
    store = get_store()
    try:
        store._validate_name(name)
    except ValueError as exc:
        return str(exc)

    # If no value provided, prompt interactively
    if value is None:
        if cli is not None:
            # Use the CLI's secret prompt if available
            try:
                from hermes_cli.secret_prompt import masked_secret_prompt
                value = masked_secret_prompt(f"Enter value for '{name}' (hidden): ")
            except (ImportError, Exception):
                import getpass
                try:
                    value = getpass.getpass(f"Enter value for '{name}' (hidden): ")
                except (EOFError, KeyboardInterrupt):
                    value = None
        else:
            # No CLI context - try getpass directly
            import getpass
            try:
                value = getpass.getpass(f"Enter value for '{name}' (hidden): ")
            except (EOFError, KeyboardInterrupt):
                value = None

        if not value:
            return "No value provided. Credential not stored."

    result = store.set(name, value.strip())
    if result["success"]:
        # Register with redactor
        try:
            from agent.redact import register_secret
            register_secret(value)
        except Exception:
            pass
        if result.get("created"):
            return f"Stored '{name}'."
        return f"Updated '{name}'."
    return result.get("error", "Failed to store credential.")


def _handle_creds_remove(name: str) -> str:
    """Handle /creds remove <name> - delete a credential."""
    if not name:
        return "Usage: /creds remove <name>"

    result = get_store().delete(name.strip())
    if result["success"]:
        return f"Removed '{name}'."
    return result.get("error", f"Failed to remove '{name}'.")


def _handle_creds_clear() -> str:
    """Handle /creds clear - remove all credentials."""
    store = get_store()
    entries = store.list()
    count = len(entries)
    for entry in entries:
        store.delete(entry["name"])
    return f"Cleared all {count} credential(s)."


def _handle_creds_inject(arg: str) -> str:
    """Handle /creds inject <name> - make a credential available as an env var."""
    if not arg:
        return "Usage: /creds inject <name1> [name2 ...]"

    names = arg.strip().split()
    store = get_store()
    injected = []
    missing = []

    for name in names:
        value = store.get(name)
        if value is None:
            missing.append(name)
            continue
        # Set the env var for the current process
        import os
        os.environ[name.upper()] = value
        # Register with redactor
        try:
            from agent.redact import register_secret
            register_secret(value)
        except Exception:
            pass
        injected.append(name)

    lines = []
    if injected:
        lines.append(f"Injected ({len(injected)}): {', '.join(injected)}")
        lines.append("Available as env vars in the next terminal() call.")
    if missing:
        lines.append(f"Not found: {', '.join(missing)}")

    return "\n".join(lines) if lines else "No credentials specified."


# -- Shortcut: /creds set without the 'set' prefix for quick inline storage --
# This is not used as a handler but documents the pattern for process_command():
# when args match "name=value" or just "name value", default to set.
_default_to_set_pattern = True
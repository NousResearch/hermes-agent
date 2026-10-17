"""CLI handlers for the ``/creds`` slash command.

Subcommands operate on the profile's encrypted credential store
(``agent/credential_store.py``). ``show`` reveals a plaintext value, so the
command is registered ``cli_only``: a messaging surface must never echo a
secret into a chat (and from there back into model context). Values that leave
the store are registered with the redactor so later output paths scrub them.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Callable, Dict, List, Optional

from agent.credential_store import (
    STORE_FILE,
    CredentialStore,
    CredentialStoreError,
    get_store,
    validate_name,
)

logger = logging.getLogger(__name__)

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Names whose injection would redirect or hijack this process and everything it
# spawns (PATH/LD_PRELOAD-style), or that Hermes already manages itself.
_FORBIDDEN_ENV_NAMES = frozenset({
    "PATH", "HOME", "SHELL", "USER", "LOGNAME", "PWD", "TMPDIR", "IFS",
    "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT",
    "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH",
    "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT",
    "BASH_ENV", "ENV", "NODE_OPTIONS", "PERL5OPT", "SHELLOPTS",
})

_USAGE = """Usage:
  /creds                       List stored credential names
  /creds list                  List stored credential names
  /creds set <name> [value]    Store a credential (masked prompt if value omitted;
                               --force replaces an existing one)
  /creds show <name>           Print a credential's value (this terminal only)
  /creds remove <name>         Delete a credential (alias: delete)
  /creds inject <name> ...     Export credential(s) as env vars for this CLI session
  /creds unset <name> ...      Remove previously injected env vars
  /creds clear --yes           Delete every stored credential (needs --yes)"""


def handle_creds_command(args: str) -> str:
    """Process a /creds invocation; returns the text the surface should show."""
    text = (args or "").strip()
    if not text:
        return _handle_list()
    subcommand, _, rest = text.partition(" ")
    handlers: dict[str, Callable[[], str]] = {
        "list": _handle_list,
        "ls": _handle_list,
        "show": lambda: _handle_show(rest),
        "set": lambda: _handle_set(rest),
        "remove": lambda: _handle_remove(rest),
        "delete": lambda: _handle_remove(rest),
        "clear": lambda: _handle_clear(rest),
        "inject": lambda: _handle_inject(rest),
        "unset": lambda: _handle_unset(rest),
        "help": lambda: _USAGE,
    }
    handler = handlers.get(subcommand.lower())
    if handler is None:
        return f"Unknown subcommand: /creds {subcommand}\n\n{_USAGE}"
    try:
        return handler()
    except CredentialStoreError as exc:
        return _store_error(exc)


def _store_error(exc: CredentialStoreError) -> str:
    """An unreadable store must never read as an empty one: say so, change nothing."""
    return (f"Credential store unavailable: {exc}\n"
            "Nothing was changed. Restore the file from a backup (or move it aside) and retry.")


def _register(value: str) -> None:
    """Register a value that just left the store so output paths scrub it."""
    from agent.redact import register_secret

    register_secret(value)


def _handle_list() -> str:
    entries = get_store().list()
    if not entries:
        return "No credentials stored. Use /creds set <name> [value] to store one."
    lines = [f"Stored credentials ({len(entries)}):", ""]
    lines.extend(f"  {entry['name']}" for entry in entries)
    lines += ["", "Use /creds show <name> to reveal a value, or pass the name in "
                  "execute_code's `credentials` parameter."]
    return "\n".join(lines)


def _handle_show(arg: str) -> str:
    name = arg.strip()
    if not name:
        return "Usage: /creds show <name>"
    value = get_store().get(name)
    if value is None:
        return f"Credential '{name}' not found."
    # Registered AFTER the reply text is built: the user asked to see it here,
    # and any later model-facing echo of the same bytes gets scrubbed.
    _register(value)
    return f"{name}: {value}"


def _prompt_value(name: str) -> Optional[str]:
    """Masked interactive prompt; None when cancelled or stdin is not usable."""
    prompt = f"Enter value for '{name}' (hidden): "
    try:
        from hermes_cli.secret_prompt import masked_secret_prompt

        return masked_secret_prompt(prompt) or None
    except (EOFError, KeyboardInterrupt):
        return None
    except ImportError:
        import getpass

        try:
            return getpass.getpass(prompt) or None
        except (EOFError, KeyboardInterrupt):
            return None


def _handle_set(arg: str) -> str:
    text = arg.strip()
    if not text:
        return "Usage: /creds set <name> [value] [--force]\n  Omit the value for a masked prompt."
    force = False
    for flag in ("--force", "-f"):
        if re.search(rf"(^|\s){re.escape(flag)}(?=\s|$)", text):
            force = True
            text = re.sub(rf"\s*{re.escape(flag)}", "", text, count=1)
    parts = text.split(None, 1)
    name = parts[0]
    value = parts[1].strip() if len(parts) > 1 else None
    try:
        validate_name(name)
    except ValueError as exc:
        return str(exc)
    if value is None:
        value = _prompt_value(name)
        if not value:
            return "Cancelled — nothing stored."
    store = get_store()
    result = store.set(name, value, overwrite=force)
    if not result.get("success"):
        if "already exists" in str(result.get("error", "")):
            return (f"{result['error']}\n"
                    f"Update it with: /creds set {name} <value> --force")
        return str(result.get("error") or "Failed to store credential.")
    _register(value)
    verb = "Updated" if result.get("updated") else "Stored"
    return f"{verb} '{name}' (encrypted in {store.base_dir / STORE_FILE})."


def _handle_remove(arg: str) -> str:
    name = arg.strip()
    if not name:
        return "Usage: /creds remove <name>"
    result = get_store().delete(name)
    if result.get("success"):
        return f"Removed '{name}'."
    return str(result.get("error") or f"Failed to remove '{name}'.")


def _handle_clear(arg: str) -> str:
    if "--yes" not in arg.split():
        count = len(get_store().list())
        return (f"Refusing to delete {count} credential(s) without confirmation.\n"
                "Run /creds clear --yes to delete them all.")
    removed = get_store().clear()["removed"]
    return f"Cleared {removed} credential(s)."


def _child_would_see(name: str) -> bool:
    """True when the terminal/execute_code env policy passes this name to children."""
    try:
        from tools.environments.local_env_policy import (
            _ALWAYS_STRIP_FOLDED,
            _HERMES_PROVIDER_ENV_BLOCKLIST,
        )
    except ImportError:  # pragma: no cover - policy module ships with the runtime
        return True
    upper = name.upper()
    return (name not in _HERMES_PROVIDER_ENV_BLOCKLIST
            and upper not in _HERMES_PROVIDER_ENV_BLOCKLIST
            and upper not in _ALWAYS_STRIP_FOLDED)


def _handle_inject(arg: str) -> str:
    names = [token for token in arg.split() if not token.startswith("-")]
    if not names:
        return "Usage: /creds inject <name> [name2 ...]"
    store: CredentialStore = get_store()
    env_names: list[str] = []
    injected: list[str] = []
    missing: list[str] = []
    stripped: list[str] = []
    refused: list[str] = []
    for name in names:
        upper = name.upper()
        if not _ENV_NAME_RE.match(upper):
            refused.append(f"{name} (not a valid env var name — no hyphens/spaces)")
            continue
        if upper in _FORBIDDEN_ENV_NAMES:
            refused.append(f"{name} → {upper} (refused: injecting it would redirect this process)")
            continue
        value = store.get(name)
        if value is None:
            missing.append(name)
            continue
        os.environ[upper] = value
        _register(value)
        env_names.append(upper)
        injected.append(name)
        if not _child_would_see(name):
            stripped.append(upper)
    lines: list[str] = []
    if injected:
        lines.append(f"Injected ({len(injected)}): {', '.join(env_names)}")
        lines.append("Set in this CLI process, so terminal() and execute_code children inherit "
                     "them (names Hermes strips from child envs are noted below; unset with "
                     "/creds unset).")
    if stripped:
        lines.append(f"Note: {', '.join(stripped)} are Hermes-managed names and are stripped from "
                     "child process environments — those secrets belong in config.yaml instead.")
    if missing:
        lines.append(f"Not found: {', '.join(missing)}")
    if refused:
        lines.append("Refused: " + "; ".join(refused))
    return "\n".join(lines) if lines else "No credentials specified."


def _handle_unset(arg: str) -> str:
    names = [token for token in arg.split() if not token.startswith("-")]
    if not names:
        return "Usage: /creds unset <name> [name2 ...]"
    removed: list[str] = []
    absent: list[str] = []
    for name in names:
        upper = name.upper()
        if upper in os.environ:
            os.environ.pop(upper, None)
            removed.append(upper)
        else:
            absent.append(upper)
    lines = []
    if removed:
        lines.append(f"Unset ({len(removed)}): {', '.join(removed)}")
    if absent:
        lines.append(f"Not set: {', '.join(absent)}")
    return "\n".join(lines)
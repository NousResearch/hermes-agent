"""Terminal-only capture and management of secrets stored in the active profile's ``.env``."""

from __future__ import annotations

import getpass
import sys


def _error(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def _read_secret(*, from_stdin: bool) -> str | None:
    if from_stdin:
        if sys.stdin.isatty():
            _error("--stdin requires redirected input; omit it to use the hidden prompt.")
            return None
        value = sys.stdin.read().rstrip("\r\n")
    else:
        if not sys.stdin.isatty():
            _error("a terminal is required for hidden input; pipe the value and pass --stdin.")
            return None
        try:
            value = getpass.getpass("Secret value (hidden): ")
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            _error("secret input was cancelled.")
            return None
    if "\0" in value:
        _error("secret value cannot contain NUL bytes.")
        return None
    if "\r" in value or "\n" in value:
        _error("secret value must be a single line.")
        return None
    if not value:
        _error("secret value cannot be empty.")
        return None
    return value


def cmd_set(args) -> int:
    """Capture a value without accepting it in argv, then use the shared credential lifecycle."""
    from hermes_cli.config import get_env_path, save_env_value_secure, validate_env_var_name_for_write

    if args._unexpected:
        _error("secret values must use hidden input or redirected stdin, never command arguments.")
        return 2
    name = args.name
    try:
        validate_env_var_name_for_write(name)
    except ValueError as exc:
        return _error(str(exc))
    value = _read_secret(from_stdin=bool(args.stdin))
    if value is None:
        return 1
    if not value.isascii():
        return _error("secret value must contain ASCII characters only.")
    try:
        result = save_env_value_secure(name, value)
    except (RuntimeError, ValueError) as exc:
        return _error(str(exc))
    if not result.get("success"):
        return 1
    print(f"Stored {name} in {get_env_path()} (value hidden).")
    return 0


def cmd_list(_args) -> int:
    """List local ``.env`` names only; values never cross the output boundary."""
    from hermes_cli.config import load_env

    names = sorted(load_env())
    if not names:
        print("No local secrets are configured for this profile.")
        return 0
    print("Local secret names (values hidden):")
    for name in names:
        print(f"  {name}")
    return 0


def cmd_delete(args) -> int:
    """Delete through the shared lifecycle so provider pools and credential mirrors are pruned."""
    from hermes_cli.credential_lifecycle import remove_provider_env_credential

    name = args.name
    try:
        result = remove_provider_env_credential(name)
    except (RuntimeError, ValueError) as exc:
        return _error(str(exc))
    if not result.get("ok"):
        return 1
    if not result.get("found"):
        return _error(f"{name} is not configured for this profile.")
    print(f"Deleted {name} (value hidden).")
    return 0

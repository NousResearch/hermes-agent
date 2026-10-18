"""CLI handlers for the ``/creds`` slash command.

Subcommands operate on the profile's credential store
(``agent/credential_store.py``) plus its platform backends
(``agent/credential_backends.py``): a value is written to Bitwarden Secrets
Manager or 1Password when that integration is available, and the local store
keeps only an index entry for it — otherwise the local encrypted store is used
exactly as before. ``show`` reveals a plaintext value, so the command is
registered ``cli_only``: a messaging surface must never echo a secret into a
chat (and from there back into model context). Values that leave the store are
registered with the redactor so later output paths scrub them.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Callable, Dict, List, Optional

from agent.credential_backends import (
    LOCAL,
    AmbiguousBackendError,
    BackendError,
    format_statuses,
    get_backend,
    remove_credential,
    save_credential,
)
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
  /creds list                  List stored credential names (shows where each lives)
  /creds set <name> [value]    Store a credential (masked prompt if value omitted;
                               --force replaces an existing one; --backend <name>
                               or --local chooses the target instead of the prompt)
  /creds show <name>           Print a credential's value (this terminal only)
  /creds remove <name>         Delete a credential, including from its platform
  /creds inject <name> ...     Export credential(s) as env vars for this CLI session
  /creds unset <name> ...      Remove previously injected env vars
  /creds status                Show which secret platforms are available
  /creds clear --yes           Delete every stored credential (needs --yes)

Backends: local (always), plus bitwarden / onepassword when `hermes secrets`
has set them up. With more than one available, /creds asks where to store."""


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
        "status": _handle_status,
        "backends": _handle_status,
        "help": lambda: _USAGE,
    }
    handler = handlers.get(subcommand.lower())
    if handler is None:
        return f"Unknown subcommand: /creds {subcommand}\n\n{_USAGE}"
    try:
        return handler()
    except CredentialStoreError as exc:
        return _store_error(exc)
    except BackendError as exc:
        return str(exc)


def _store_error(exc: CredentialStoreError) -> str:
    """An unreadable store must never read as an empty one: say so, change nothing."""
    return (f"Credential store unavailable: {exc}\n"
            "Nothing was changed. Restore the file from a backup (or move it aside) and retry.")


def _register(value: str) -> None:
    """Register a value that just left the store so output paths scrub it."""
    from agent.redact import register_secret

    register_secret(value)


def _backend_label(name: str) -> str:
    return "Local encrypted store" if name == LOCAL else get_backend(name).display_name


def _handle_list() -> str:
    entries = get_store().list()
    if not entries:
        return "No credentials stored. Use /creds set <name> [value] to store one."
    width = max(len(entry["name"]) for entry in entries)
    lines = [f"Stored credentials ({len(entries)}):", ""]
    lines.extend(f"  {entry['name']:<{width}}  [{entry.get('backend', LOCAL)}]"
                 for entry in entries)
    lines += ["", "Use /creds show <name> to reveal a value, or pass the name in "
                  "execute_code's `credentials` parameter."]
    return "\n".join(lines)


def _handle_status() -> str:
    """Which platforms can take a new credential right now (and why not)."""
    return format_statuses()


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


def _extract_flags(text: str) -> tuple[str, bool, Optional[str]]:
    """Pull ``--force``/``-f``, ``--backend <name>``/``-b <name>`` and ``--local``
    out of the argument string, leaving name + value untouched (value keeps its
    internal spacing). Returns ``(rest, force, backend)``."""
    force = False
    backend: Optional[str] = None
    match = re.search(r"(?:^|\s)--backend\s+(\S+)", text)
    if match:
        backend = match.group(1)
        text = text[:match.start()] + text[match.end():]
    if backend is None:
        match = re.search(r"(?:^|\s)-b\s+(\S+)", text)
        if match:
            backend = match.group(1)
            text = text[:match.start()] + text[match.end():]
    if re.search(r"(?:^|\s)--local(?:\s|$)", text):
        backend = LOCAL
        text = re.sub(r"\s*--local", "", text, count=1)
    for flag in ("--force", "-f"):
        if re.search(rf"(?:^|\s){re.escape(flag)}(?=\s|$)", text):
            force = True
            text = re.sub(rf"\s*{re.escape(flag)}", "", text, count=1)
    return text.strip(), force, backend


def _choose_backend(names: list[str]) -> Optional[str]:
    """Numbered picker over the available platforms plus local.

    Returns None when the terminal cannot answer (piped stdin) or the user
    cancels — the caller reports that ``--backend`` decides it instead.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    options = [*names, LOCAL]
    for index, name in enumerate(options, 1):
        print(f"  {index}) {_backend_label(name)}")
    while True:
        try:
            answer = input(f"Store where? [1-{len(options)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not answer:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        print(f"  Enter a number 1-{len(options)}.")


def _save(name: str, value: str, *, backend: Optional[str], force: bool) -> dict:
    """save_credential with the picker wired in for an ambiguous ``auto`` target."""
    requested = backend or "auto"
    try:
        return save_credential(name, value, backend=requested, overwrite=force)
    except AmbiguousBackendError as exc:
        chosen = _choose_backend(exc.names)
        if not chosen:
            return {"success": False, "error":
                    f"More than one secret platform is available ({', '.join(exc.names)}).\n"
                    f"Re-run with --backend <name> (or --local) to choose one."}
        return save_credential(name, value, backend=chosen, overwrite=force)


def _handle_set(arg: str) -> str:
    text, force, backend = _extract_flags(arg.strip())
    if not text:
        return ("Usage: /creds set <name> [value] [--force] [--backend <name>|--local]\n"
                "  Omit the value for a masked prompt.")
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
    result = _save(name, value, backend=backend, force=force)
    if not result.get("success"):
        error = str(result.get("error") or "Failed to store credential.")
        # save_credential already names --force; only the store's own wording
        # ("Use overwrite to replace it") needs the actionable hint appended.
        if "already exists" in error and "--force" not in error:
            error = f"{error}\nUpdate it with: /creds set {name} <value> --force"
        return error
    _register(value)
    where = _where_phrase(str(result.get("backend", LOCAL)))
    verb = "Updated" if result.get("updated") else "Stored"
    lines = [f"{verb} '{name}' {where}."]
    lines.extend(f"Note: {warning}" for warning in result.get("warnings", []))
    return "\n".join(lines)


def _where_phrase(backend_name: str) -> str:
    if backend_name == LOCAL:
        return f"in the local encrypted store ({get_store().base_dir / STORE_FILE})"
    return f"in {_backend_label(backend_name)}"


def _handle_remove(arg: str) -> str:
    name = arg.strip()
    if not name:
        return "Usage: /creds remove <name>"
    result = remove_credential(name)
    if not result.get("success"):
        return str(result.get("error") or f"Failed to remove '{name}'.")
    backend_name = str(result.get("backend") or LOCAL)
    if backend_name == LOCAL:
        return f"Removed '{name}'."
    return f"Removed '{name}' from {_backend_label(backend_name)}."


def _handle_clear(arg: str) -> str:
    if "--yes" not in arg.split():
        count = len(get_store().list())
        return (f"Refusing to delete {count} credential(s) without confirmation.\n"
                "Run /creds clear --yes to delete them all.")
    removed = 0
    failures: list[str] = []
    for entry in get_store().list():
        result = remove_credential(entry["name"])
        if result.get("success"):
            removed += 1
        else:
            failures.append(str(result.get("error") or f"Failed to remove {entry['name']}."))
    lines = [f"Cleared {removed} credential(s).", *failures]
    return "\n".join(lines)


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
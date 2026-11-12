"""Launch flags that become process environment before any startup config read.

Split out of ``hermes_cli/main.py``. Each value set here is either a process-wide opt-in
(``--safe-mode``, ``--ignore-user-config``) or a behaviour value the gateway reads per session from
the launching shell's ``client_env`` (``--append-system-prompt``; see ``agent.session_client_env``).
"""

import os
import sys

APPEND_PROMPT_ENV = "HERMES_APPEND_SYSTEM_PROMPT"


def _apply_safe_mode(args) -> None:
    if not getattr(args, "safe_mode", False):
        return
    os.environ["HERMES_SAFE_MODE"] = "1"
    os.environ["HERMES_IGNORE_USER_CONFIG"] = "1"
    os.environ["HERMES_IGNORE_RULES"] = "1"


def _apply_user_config_bypass(args) -> None:
    """Apply the explicit config bypass before any startup config reads."""
    if getattr(args, "ignore_user_config", False):
        os.environ["HERMES_IGNORE_USER_CONFIG"] = "1"


def _apply_append_system_prompt(args) -> None:
    """``--append-system-prompt TEXT`` / ``--append-system-prompt-file PATH`` (both may be given; text
    first): this launch's addition after the session's system-prompt overlay. It rides the new
    session's ``client_env``, so it reaches only the sessions this process creates, never the shared
    gateway's other sessions."""
    parts = [getattr(args, "append_system_prompt", None) or ""]
    path = getattr(args, "append_system_prompt_file", None)
    if path:
        try:
            with open(os.path.expanduser(path), "r", encoding="utf-8-sig", errors="replace") as fh:
                parts.append(fh.read())
        except OSError as exc:
            print(f"Error: cannot read --append-system-prompt-file {path}: {exc}", file=sys.stderr)
            sys.exit(2)
    text = "\n\n".join(part.strip() for part in parts if part and part.strip())
    if not text:
        return
    os.environ[APPEND_PROMPT_ENV] = text
    if getattr(args, "resume", None) or getattr(args, "continue_last", None):
        # A session's prompt additions are frozen at creation, like -s skills.
        print("Note: --append-system-prompt applies only if this launch creates a new session; "
              "a resumed session keeps the prompt it was created with.", file=sys.stderr)

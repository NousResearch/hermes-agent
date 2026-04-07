"""Interactive prompt callbacks for terminal_tool integration.

These bridge terminal_tool's interactive prompts (clarify, sudo, approval)
into prompt_toolkit's event loop. Each function takes the HermesCLI instance
as its first argument and uses its state (queues, app reference) to coordinate
with the TUI.
"""

import queue
import time as _time
import getpass

from hermes_cli.banner import cprint, _DIM, _RST
from hermes_cli.config import save_env_value_secure
from hermes_constants import display_hermes_home


def clarify_callback(cli, question, choices):
    """Prompt for clarifying question through the TUI.

    Sets up the interactive selection UI, then blocks until the user
    responds. Returns the user's choice or a timeout message.
    """
    from cli import CLI_CONFIG

    timeout = CLI_CONFIG.get("clarify", {}).get("timeout", 120)
    response_queue = queue.Queue()
    is_open_ended = not choices

    cli._clarify_state = {
        "question": question,
        "choices": choices if not is_open_ended else [],
        "selected": 0,
        "response_queue": response_queue,
    }
    cli._clarify_deadline = _time.monotonic() + timeout
    cli._clarify_freetext = is_open_ended

    if hasattr(cli, "_app") and cli._app:
        cli._app.invalidate()

    while True:
        try:
            result = response_queue.get(timeout=1)
            cli._clarify_deadline = 0
            return result
        except queue.Empty:
            remaining = cli._clarify_deadline - _time.monotonic()
            if remaining <= 0:
                break
            if hasattr(cli, "_app") and cli._app:
                cli._app.invalidate()

    cli._clarify_state = None
    cli._clarify_freetext = False
    cli._clarify_deadline = 0
    if hasattr(cli, "_app") and cli._app:
        cli._app.invalidate()
    cprint(f"\n{_DIM}(clarify timed out after {timeout}s — agent will decide){_RST}")
    return (
        "The user did not provide a response within the time limit. "
        "Use your best judgement to make the choice and proceed."
    )


def sudo_password_callback(cli) -> str:
    """Prompt for sudo password through the TUI.

    Sets up a password input area and blocks until the user responds.
    """
    timeout = 45
    response_queue = queue.Queue()

    cli._sudo_state = {"response_queue": response_queue}
    cli._sudo_deadline = _time.monotonic() + timeout

    if hasattr(cli, "_app") and cli._app:
        cli._app.invalidate()

    while True:
        try:
            result = response_queue.get(timeout=1)
            cli._sudo_state = None
            cli._sudo_deadline = 0
            if hasattr(cli, "_app") and cli._app:
                cli._app.invalidate()
            if result:
                cprint(f"\n{_DIM}  ✓ Password received (cached for session){_RST}")
            else:
                cprint(f"\n{_DIM}  ⏭ Skipped{_RST}")
            return result
        except queue.Empty:
            remaining = cli._sudo_deadline - _time.monotonic()
            if remaining <= 0:
                break
            if hasattr(cli, "_app") and cli._app:
                cli._app.invalidate()

    cli._sudo_state = None
    cli._sudo_deadline = 0
    if hasattr(cli, "_app") and cli._app:
        cli._app.invalidate()
    cprint(f"\n{_DIM}  ⏱ Timeout — continuing without sudo{_RST}")
    return ""


def prompt_for_secret(cli, var_name: str, prompt: str, metadata=None) -> dict:
    """Prompt for a secret value through the TUI (e.g. API keys for skills).

    Returns a dict with keys: success, stored_as, validated, skipped, message.
    The secret is stored in ~/.hermes/.env and never exposed to the model.
    """
    if not getattr(cli, "_app", None):
        if not hasattr(cli, "_secret_state"):
            cli._secret_state = None
        if not hasattr(cli, "_secret_deadline"):
            cli._secret_deadline = 0
        try:
            value = getpass.getpass(f"{prompt} (hidden, Enter to skip): ")
        except (EOFError, KeyboardInterrupt):
            value = ""

        if not value:
            cprint(f"\n{_DIM}  ⏭ Secret entry cancelled{_RST}")
            return {
                "success": True,
                "reason": "cancelled",
                "stored_as": var_name,
                "validated": False,
                "skipped": True,
                "message": "Secret setup was skipped.",
            }

        stored = save_env_value_secure(var_name, value)
        _dhh = display_hermes_home()
        cprint(f"\n{_DIM}  ✓ Stored secret in {_dhh}/.env as {var_name}{_RST}")
        return {
            **stored,
            "skipped": False,
            "message": "Secret stored securely. The secret value was not exposed to the model.",
        }

    timeout = 120
    response_queue = queue.Queue()

    cli._secret_state = {
        "var_name": var_name,
        "prompt": prompt,
        "metadata": metadata or {},
        "response_queue": response_queue,
    }
    cli._secret_deadline = _time.monotonic() + timeout
    # Avoid storing stale draft input as the secret when Enter is pressed.
    if hasattr(cli, "_clear_secret_input_buffer"):
        try:
            cli._clear_secret_input_buffer()
        except Exception:
            pass
    elif hasattr(cli, "_app") and cli._app:
        try:
            cli._app.current_buffer.reset()
        except Exception:
            pass

    if hasattr(cli, "_app") and cli._app:
        cli._app.invalidate()

    while True:
        try:
            value = response_queue.get(timeout=1)
            cli._secret_state = None
            cli._secret_deadline = 0
            if hasattr(cli, "_app") and cli._app:
                cli._app.invalidate()

            if not value:
                cprint(f"\n{_DIM}  ⏭ Secret entry cancelled{_RST}")
                return {
                    "success": True,
                    "reason": "cancelled",
                    "stored_as": var_name,
                    "validated": False,
                    "skipped": True,
                    "message": "Secret setup was skipped.",
                }

            stored = save_env_value_secure(var_name, value)
            _dhh = display_hermes_home()
            cprint(f"\n{_DIM}  ✓ Stored secret in {_dhh}/.env as {var_name}{_RST}")
            return {
                **stored,
                "skipped": False,
                "message": "Secret stored securely. The secret value was not exposed to the model.",
            }
        except queue.Empty:
            remaining = cli._secret_deadline - _time.monotonic()
            if remaining <= 0:
                break
            if hasattr(cli, "_app") and cli._app:
                cli._app.invalidate()

    cli._secret_state = None
    cli._secret_deadline = 0
    if hasattr(cli, "_clear_secret_input_buffer"):
        try:
            cli._clear_secret_input_buffer()
        except Exception:
            pass
    elif hasattr(cli, "_app") and cli._app:
        try:
            cli._app.current_buffer.reset()
        except Exception:
            pass
    if hasattr(cli, "_app") and cli._app:
        cli._app.invalidate()
    cprint(f"\n{_DIM}  ⏱ Timeout — secret capture cancelled{_RST}")
    return {
        "success": True,
        "reason": "timeout",
        "stored_as": var_name,
        "validated": False,
        "skipped": True,
        "message": "Secret setup timed out and was skipped.",
    }


def approval_callback(cli, command: str, description: str) -> str:
    """Prompt for dangerous command approval through the TUI.

    Shows a selection UI with choices: once / session / always / deny.
    When the command is longer than 70 characters, a "view" option is
    included so the user can reveal the full text before deciding.

    Uses cli._approval_lock to serialize concurrent requests (e.g. from
    parallel delegation subtasks) so each prompt gets its own turn.
    """
    lock = getattr(cli, "_approval_lock", None)
    if lock is None:
        import threading
        cli._approval_lock = threading.Lock()
        lock = cli._approval_lock

    with lock:
        from cli import CLI_CONFIG
        timeout = CLI_CONFIG.get("approvals", {}).get("timeout", 60)
        response_queue = queue.Queue()
        choices = ["once", "session", "always", "deny"]
        if len(command) > 70:
            choices.append("view")

        cli._approval_state = {
            "command": command,
            "description": description,
            "choices": choices,
            "selected": 0,
            "response_queue": response_queue,
        }
        cli._approval_deadline = _time.monotonic() + timeout

        if hasattr(cli, "_app") and cli._app:
            cli._app.invalidate()

        while True:
            try:
                result = response_queue.get(timeout=1)
                cli._approval_state = None
                cli._approval_deadline = 0
                if hasattr(cli, "_app") and cli._app:
                    cli._app.invalidate()
                return result
            except queue.Empty:
                remaining = cli._approval_deadline - _time.monotonic()
                if remaining <= 0:
                    break
                if hasattr(cli, "_app") and cli._app:
                    cli._app.invalidate()

        cli._approval_state = None
        cli._approval_deadline = 0
        if hasattr(cli, "_app") and cli._app:
            cli._app.invalidate()
        cprint(f"\n{_DIM}  ⏱ Timeout — denying command{_RST}")
        return "deny"

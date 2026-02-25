"""Interactive prompt callbacks for terminal_tool integration.

These bridge terminal_tool's interactive prompts (clarify, sudo, approval)
into prompt_toolkit's event loop. Each function takes the HermesCLI instance
as its first argument and uses its state (queues, app reference) to coordinate
with the TUI.
"""

import queue
import time as _time

from hermes_cli.banner import _DIM, _RST, cprint


def clarify_callback(cli, question, choices):
    """Prompt for clarifying question through the TUI.

    Sets up the interactive selection UI, then blocks until the user
    responds. Returns the user's choice or a timeout message.
    """
    from cli import CLI_CONFIG

    timeout = CLI_CONFIG.get("clarify", {}).get("timeout", 120)
    response_queue = queue.Queue()
    is_open_ended = not choices or len(choices) == 0

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


def approval_callback(cli, command: str, description: str) -> str:
    """Prompt for dangerous command approval through the TUI.

    Shows a selection UI with choices: once / session / always / deny.
    """
    timeout = 60
    response_queue = queue.Queue()
    choices = ["once", "session", "always", "deny"]

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

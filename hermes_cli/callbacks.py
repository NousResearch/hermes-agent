"""Secret-capture prompt for the interactive CLI (``_secret_capture_callback`` backend)."""

import queue
import time as _time

from hermes_cli.banner import cprint, _DIM, _RST
from hermes_cli.config import save_env_value_secure
from hermes_cli.secret_prompt import masked_secret_prompt
from hermes_constants import display_hermes_home


def _invalidate(cli) -> None:
    if getattr(cli, "_app", None):
        cli._app.invalidate()


def _clear_secret_input(cli) -> None:
    """Drop stale draft input so Enter never stores it as the secret."""
    try:
        if hasattr(cli, "_clear_secret_input_buffer"):
            cli._clear_secret_input_buffer()
        elif getattr(cli, "_app", None):
            cli._app.current_buffer.reset()
    except Exception:
        pass


def _skipped(var_name: str, reason: str, message: str) -> dict:
    return {
        "success": True, "reason": reason, "stored_as": var_name,
        "validated": False, "skipped": True, "message": message}


def _secret_result(var_name: str, value: str) -> dict:
    """Store ``value`` (or report a skip when empty) and build the callback result dict."""
    if not value:
        cprint(f"\n{_DIM}  ⏭ Secret entry skipped{_RST}")
        return _skipped(var_name, "cancelled", "Secret setup was skipped.")
    stored = save_env_value_secure(var_name, value)
    cprint(f"\n{_DIM}  ✓ Stored secret in {display_hermes_home()}/.env as {var_name}{_RST}")
    return {
        **stored,
        "skipped": False,
        "message": "Secret stored securely. The secret value was not exposed to the model."}


def prompt_for_secret(cli, var_name: str, prompt: str, metadata=None) -> dict:
    """Prompt for a secret value through the TUI (e.g. API keys for skills).

    Returns a dict with keys: success, stored_as, validated, skipped, message. The secret is stored
    in ~/.hermes/.env and never exposed to the model.
    """
    if not getattr(cli, "_app", None):
        if not hasattr(cli, "_secret_state"):
            cli._secret_state = None
        if not hasattr(cli, "_secret_deadline"):
            cli._secret_deadline = 0
        try:
            value = masked_secret_prompt(f"{prompt} (hidden, ESC or empty Enter to skip): ")
        except (EOFError, KeyboardInterrupt):
            value = ""
        return _secret_result(var_name, value)

    response_queue = queue.Queue()
    cli._secret_state = {
        "var_name": var_name,
        "prompt": prompt,
        "metadata": metadata or {},
        "response_queue": response_queue}
    cli._secret_deadline = _time.monotonic() + 120
    if hasattr(cli, "_ring_bell"):
        cli._ring_bell(prompt=True, context=f"secret needed ({var_name})")
    _clear_secret_input(cli)
    _invalidate(cli)

    while True:
        try:
            value = response_queue.get(timeout=1)
        except queue.Empty:
            if cli._secret_deadline - _time.monotonic() <= 0:
                break
            _invalidate(cli)
            continue
        cli._secret_state = None
        cli._secret_deadline = 0
        _invalidate(cli)
        return _secret_result(var_name, value)

    cli._secret_state = None
    cli._secret_deadline = 0
    _clear_secret_input(cli)
    _invalidate(cli)
    cprint(f"\n{_DIM}  ⏱ Timeout — secret capture cancelled{_RST}")
    return _skipped(var_name, "timeout", "Secret setup timed out and was skipped.")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

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
        timeout = CLI_CONFIG.get("approvals", {}).get("timeout", 300)
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
        return "timeout"

def clarify_callback(cli, question, choices, multi_select=False):
    """Prompt for clarifying question through the TUI.

    Sets up the interactive selection UI, then blocks until the user
    responds. Returns the user's choice or a timeout message.

    When ``multi_select`` is True, shows checkboxes and the user can
    select multiple options with Space, confirming with Enter.
    """
    from cli import CLI_CONFIG
    from tools.clarify_gateway import resolve_clarify_timeout

    # Canonical clarify timeout, shared with the gateway/TUI path. `<= 0`
    # means unlimited (never auto-skip mid-think) → a null deadline.
    timeout = resolve_clarify_timeout(CLI_CONFIG)
    response_queue = queue.Queue()
    is_open_ended = not choices
    effective_multi = multi_select and not is_open_ended

    cli._clarify_state = {
        "question": question,
        "choices": choices if not is_open_ended else [],
        "selected": 0,
        "multi_select": effective_multi,
        "selected_indices": set() if effective_multi else None,
        "response_queue": response_queue,
    }
    cli._clarify_deadline = None if timeout <= 0 else _time.monotonic() + timeout
    cli._clarify_freetext = is_open_ended

    if hasattr(cli, "_app") and cli._app:
        cli._app.invalidate()

    while True:
        try:
            result = response_queue.get(timeout=1)
            cli._clarify_deadline = None
            return result
        except queue.Empty:
            # None deadline = unlimited: never auto-skip, just keep polling.
            if cli._clarify_deadline is not None:
                remaining = cli._clarify_deadline - _time.monotonic()
                if remaining <= 0:
                    break
            if hasattr(cli, "_app") and cli._app:
                cli._app.invalidate()

    cli._clarify_state = None
    cli._clarify_freetext = False
    cli._clarify_deadline = None
    if hasattr(cli, "_app") and cli._app:
        cli._app.invalidate()
    cprint(f"\n{_DIM}(clarify timed out after {timeout}s — agent will decide){_RST}")
    return (
        "The user did not provide a response within the time limit. "
        "Use your best judgement to make the choice and proceed."
    )
# ---- END PLUGIN-COMPAT ----

"""Terminal and chat rendering for the shared sign-in flow."""

from __future__ import annotations

import contextlib
from typing import Iterator, Optional

from hermes_cli.anon_sign_in import Code, SignInState, UPGRADE_CANCELLED, UPGRADE_START


def render_sign_in_cli_code(
    state: Code, *, open_browser: bool = False, chat: bool = True, printer=print) -> None:
    """Print the ``Code`` state alone: the link, the code, and the do-not-share line."""
    from hermes_cli.auth_device_flow import _print_device_code_instructions
    _print_device_code_instructions(
        state.link, state.code, open_browser=open_browser, swallow_open_errors=True)
    printer(f"  {state.copy_with_wait if chat else state.copy}")


def drain_sign_in_copy(gen: Iterator[SignInState], *, chat: bool = True, on_terminal=None) -> str:
    """Iterate *gen* to its terminal state, discarding ``Waiting``, and return that state's copy.

    ``on_terminal`` is called with each terminal state — the CLI uses it to move the running
    session off the free tier's model. A raising hook never costs the caller its copy.
    """
    copy = ""
    for state in gen:
        if state.terminal:
            copy = state.copy if chat else state.copy_terminal
            if on_terminal is not None:
                with contextlib.suppress(Exception):
                    on_terminal(state)
    return copy


def render_sign_in_cli(
    *, timeout_seconds: float = 15.0, open_browser: bool = False, chat: bool = False,
    states: Optional[Iterator[SignInState]] = None, printer=print) -> int:
    """Print a sign-in; returns the process exit code (0 ok, 1 not, 130 interrupted).

    ``chat=True`` renders the in-chat wording; *states* lets a caller hand in a partially drained
    :func:`run_sign_in` generator; *printer* lets a caller pin the output target.
    """
    from hermes_cli import anon_auth as _core

    gen = states if states is not None else _core.run_sign_in(timeout_seconds=timeout_seconds)
    started = False
    try:
        for state in gen:
            if not started and not state.precondition:
                printer(UPGRADE_START)
                started = True
            if state.kind == "code":
                _core.render_sign_in_cli_code(state, open_browser=open_browser, chat=chat, printer=printer)
                continue
            if state.kind == "waiting":
                if not chat:          # in a chat the wait line already rode the code
                    printer(state.copy)
                continue
            if state.terminal:
                printer(state.copy if chat else state.copy_terminal)
                return 0 if state.ok else 1
    except KeyboardInterrupt:
        with contextlib.suppress(Exception):
            gen.close()
        printer(UPGRADE_CANCELLED)
        return 130
    return 1


def upgrade_guest(args) -> int:
    """``hermes auth upgrade``: sign in with a Nous account, transferring the free tier's connectors.

    Returns 0 on success (or when already signed in), 1 otherwise, 130 on Ctrl-C. Never persists
    anything unless the transfer completed AND the token grant succeeded.
    """
    from hermes_cli import anon_auth as _core
    from hermes_cli.auth_device_flow import _is_remote_session
    timeout_seconds = float(getattr(args, "timeout", None) or 15.0)
    open_browser = not getattr(args, "no_browser", False) and not _is_remote_session()
    return _core.render_sign_in_cli(
        timeout_seconds=timeout_seconds, open_browser=open_browser, chat=False)

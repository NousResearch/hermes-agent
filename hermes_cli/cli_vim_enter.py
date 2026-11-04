"""Vim-mode Enter for the classic CLI composer (``display.vim_mode``).

In Vim INSERT (and REPLACE) mode, Enter inserts a newline at the caret and never submits;
Esc then Enter (NORMAL mode) is the explicit submit. Shared by the in-process prompt
(``cli_tui_mixin._tui_handle_enter``) and the gateway-attached ``--cli`` PromptSession
(``gateway_chat_view``) so both composers honour the setting identically.

An incomplete Vim command (operator, count, visual selection or single replacement) is
cancelled without submitting. A selected completion keeps the existing Enter path.
With Vim off, the helper leaves all input handling unchanged.
"""

from prompt_toolkit.application.current import get_app
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.vi_state import InputMode

NEWLINE = "newline"
SUBMIT = "submit"
PENDING = "pending"
_INSERT_LIKE = frozenset({InputMode.INSERT, InputMode.INSERT_MULTIPLE, InputMode.REPLACE})


def vim_enter_action(app, *, arg_present=False):
    """Classify Enter as newline, explicit submit, pending-command cancel, or existing behavior."""
    if getattr(app, "editing_mode", EditingMode.EMACS) != EditingMode.VI:
        return None
    vi, buf = app.vi_state, app.current_buffer
    if (arg_present or buf.selection_state is not None or vi.operator_func is not None
            or vi.waiting_for_digraph or vi.temporary_navigation_mode
            or vi.input_mode == InputMode.REPLACE_SINGLE):
        return PENDING
    if buf.complete_state is not None and buf.complete_state.current_completion is not None:
        return None
    if vi.input_mode in _INSERT_LIKE and not vi.temporary_navigation_mode:
        return NEWLINE
    return SUBMIT if vi.input_mode == InputMode.NAVIGATION else None


def cancel_pending_enter(app):
    """Cancel an incomplete Vim command instead of treating it as prompt submission."""
    app.vi_state.reset()
    app.vi_state.input_mode = InputMode.NAVIGATION
    app.current_buffer.exit_selection()


def vim_composer_session_kwargs(config, *, composer_active=lambda: True):
    """Extra ``PromptSession`` kwargs for the gateway-attached composer; ``{}`` with Vim off.

    The session stays single-line (Enter accepts via ``validate_and_handle``); only an INSERT
    Enter on the main composer is rebound to insert a newline. ``composer_active`` returning
    False (a modal question reusing the session) keeps Enter as a plain submit.
    Alt+Enter remains a newline. Terminals encode it like a rapid Esc, Enter, so the
    operator must enter NORMAL before pressing Enter to submit.
    """
    if not (config.get("display") or {}).get("vim_mode", False):
        return {}
    kb = KeyBindings()

    def _action():
        app = get_app()
        return vim_enter_action(app, arg_present=app.key_processor.arg is not None)

    @kb.add("enter", filter=Condition(lambda: composer_active() and _action() == NEWLINE))
    def _newline(event):
        event.current_buffer.insert_text("\n")

    @kb.add("enter", filter=Condition(lambda: composer_active() and _action() == PENDING))
    def _cancel(event):
        cancel_pending_enter(event.app)

    @kb.add("escape", "enter")
    def _modified_enter(event):
        event.current_buffer.insert_text("\n")

    return {"editing_mode": EditingMode.VI, "key_bindings": kb}

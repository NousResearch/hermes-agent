"""Vim Space-leader shortcuts for the Hermes composer (``display.vim_mode``).

In Vi NORMAL mode on a focused, editable composer with no pending count, operator, visual
selection, search or modal prompt, Space arms the leader; the next key runs one action from a
per-surface table or cancels. Escape and unknown or modified keys cancel without editing. A key
another binding owns (paste, Tab, mouse) still runs that binding and disarms the leader, as do
focus, mode and modal changes. Followers never write into the draft to invoke a command.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from prompt_toolkit.application import get_app
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters import (
    Condition,
    has_arg,
    in_paste_mode,
    is_read_only,
    is_searching,
    vi_navigation_mode,
    vi_selection_mode,
    vi_waiting_for_text_object_mode,
)
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.keys import Keys

LeaderAction = Callable[[KeyPressEvent], object]

# Keys the cancel catch-all must not swallow: their own handlers run (a paste keeps the surface's
# paste path) and the after-key hook disarms the leader.
_PASSTHROUGH_KEYS = frozenset({
    Keys.BracketedPaste, Keys.CPRResponse, Keys.Vt100MouseEvent, Keys.WindowsMouseEvent,
    Keys.ScrollUp, Keys.ScrollDown, Keys.SIGINT,
})


class VimLeaderState:
    """Leader arming, bound to the control that was focused when Space was pressed."""

    def __init__(self, blocked: Callable[[], bool] = lambda: False) -> None:
        self.blocked = blocked
        self._control: Any = None
        self._skip_next_key_hook = False
        self._hooked_processor: Any = None

    def arm(self, app: Any) -> None:
        self._control = app.layout.current_control
        # after_key_press fires for the arming Space itself; the next key is the follower.
        self._skip_next_key_hook = True
        if self._hooked_processor is not app.key_processor:
            app.key_processor.after_key_press += self._after_key_press
            self._hooked_processor = app.key_processor

    def disarm(self) -> None:
        self._control = None

    def _after_key_press(self, _sender: Any) -> None:
        if self._skip_next_key_hook:
            self._skip_next_key_hook = False
        else:
            self.disarm()

    def is_armed(self, app: Any = None) -> bool:
        if self._control is None:
            return False
        app = app or get_app()
        if (self.blocked()
                or app.layout.current_control is not self._control
                or app.vi_state.input_mode != InputMode.NAVIGATION
                or app.current_buffer.selection_state is not None):
            self.disarm()
        return self._control is not None


def _current_key_passes_through() -> bool:
    keys = get_app().key_processor.key_buffer
    return bool(keys) and keys[-1].key in _PASSTHROUGH_KEYS


def bind_vim_leader(
    kb: KeyBindings, actions: Mapping[str, LeaderAction], state: VimLeaderState,
) -> VimLeaderState:
    """Register Space-leader bindings for ``actions`` (follower key -> handler) onto ``kb``."""
    armed = Condition(lambda: state.is_armed())
    can_arm = (
        Condition(lambda: get_app().vi_state.input_mode == InputMode.NAVIGATION)
        & vi_navigation_mode
        & ~has_arg
        & ~vi_waiting_for_text_object_mode
        & ~vi_selection_mode
        & ~is_read_only
        & ~is_searching
        & ~in_paste_mode
        & Condition(lambda: not state.blocked())
        & ~armed
    )

    @kb.add(" ", filter=can_arm)
    def _arm(event: KeyPressEvent) -> None:
        state.arm(event.app)
        event.app.invalidate()

    def _cancel(event: KeyPressEvent) -> None:
        state.disarm()
        event.app.invalidate()

    def _follower(action: LeaderAction) -> LeaderAction:
        def run(event: KeyPressEvent) -> None:
            state.disarm()
            action(event)
            event.app.invalidate()
        return run

    # Eager so followers and the catch-all win over non-eager Vi keys (x, d, p, ...).
    kb.add("escape", filter=armed, eager=True)(_cancel)
    for key, action in actions.items():
        kb.add(key, filter=armed, eager=True)(_follower(action))
    kb.add(Keys.Any, filter=armed & ~Condition(_current_key_passes_through), eager=True)(_cancel)
    return state


def cli_leader_actions(cli: Any) -> dict[str, LeaderAction]:
    """Classic CLI followers: each reuses the handler its existing shortcut already runs."""
    from hermes_cli.cli_subagent_monitor import open_monitor, toggle_dock
    return {
        "p": lambda event: cli._open_command_palette(),
        "e": cli._tui_handle_open_in_editor,
        "t": lambda event: open_monitor(cli),
        "r": lambda event: toggle_dock(cli),
        "c": cli._tui_handle_ctrl_c,
        "enter": lambda event: cli._tui_submit_buffer(event, event.current_buffer),
    }


def _prefill_empty_draft(text: str) -> LeaderAction:
    def prefill(event: KeyPressEvent) -> None:
        buf = event.current_buffer
        if buf.text:
            return
        buf.text = text
        buf.cursor_position = len(text)
        event.app.vi_state.input_mode = InputMode.INSERT
    return prefill


GATEWAY_LEADER_ACTIONS: dict[str, LeaderAction] = {
    "p": _prefill_empty_draft("/"),
    "m": _prefill_empty_draft("/model "),
    "e": lambda event: event.current_buffer.open_in_editor(),
    "c": lambda event: event.current_buffer.reset(),
    "enter": lambda event: event.current_buffer.validate_and_handle(),
}


def is_vim_mode_enabled(config: Optional[dict[str, Any]] = None) -> bool:
    """Return whether ``display.vim_mode`` is enabled."""
    if config is None:
        from hermes_cli.config_effective import load_user_config_effective
        config = load_user_config_effective()
    display_cfg = config.get("display") if isinstance(config, dict) else None
    return isinstance(display_cfg, dict) and bool(display_cfg.get("vim_mode", False))


def configure_gateway_prompt_session_kwargs(
    model: str, session_id: str, config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """PromptSession kwargs for ``gateway_chat_view``; Vi mode + leader when ``display.vim_mode``."""
    vim_enabled = is_vim_mode_enabled(config)
    state = VimLeaderState()

    def _toolbar() -> str:
        base = f" \u2624 {model} \u2502 {session_id} "
        return base + "\u2502 [LEADER] " if vim_enabled and state.is_armed() else base

    kwargs: dict[str, Any] = {"erase_when_done": True, "bottom_toolbar": _toolbar}
    if vim_enabled:
        kb = KeyBindings()
        bind_vim_leader(kb, GATEWAY_LEADER_ACTIONS, state)
        kwargs["editing_mode"] = EditingMode.VI
        kwargs["key_bindings"] = kb
    return kwargs

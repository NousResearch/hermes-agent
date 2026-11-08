"""Classic-CLI ghost-text suggestion of the user's next prompt (``display.next_prompt_suggestions``).

When a turn settles, one background auxiliary call guesses the next prompt and parks it on the CLI;
``SlashCommandAutoSuggest`` renders it in the empty input and Tab accepts it. Nothing here writes
to the conversation: the suggestion is only ever sent if the user accepts it and presses Enter."""

import logging
from typing import Any, Callable, Optional

from prompt_toolkit.auto_suggest import Suggestion


logger = logging.getLogger(__name__)

_STATE_ATTR = "_next_prompt_state"


class _State:
    """Pending suggestion plus a generation counter; bumping it orphans in-flight results."""

    def __init__(self) -> None:
        self.text: Optional[str] = None
        self.generation = 0


def _state(cli: Any) -> _State:
    state = getattr(cli, _STATE_ATTR, None)
    if state is None:
        state = _State()
        setattr(cli, _STATE_ATTR, state)
    return state


def pending_suggestion(cli: Any) -> Optional[str]:
    return _state(cli).text


def clear_suggestion(cli: Any) -> None:
    """Drop the pending suggestion and orphan any in-flight call (submit / new turn)."""
    state = _state(cli)
    buf = _buffer(cli)
    if buf is not None and state.text and buf.suggestion is not None and buf.suggestion.text == state.text:
        buf.suggestion = None
    state.text = None
    state.generation += 1


def _setting_enabled() -> bool:
    from hermes_cli.config import load_config_readonly
    return bool(((load_config_readonly() or {}).get("display") or {}).get("next_prompt_suggestions"))


def _spawn(fn: Callable[[], None]) -> None:
    from agent.memory_provider import spawn_context_thread
    spawn_context_thread(fn, name="next-prompt-suggest").start()


def _buffer(cli: Any):
    app = getattr(cli, "_app", None)
    return getattr(app, "current_buffer", None)


def _idle_and_empty(cli: Any) -> bool:
    """Nothing running, nothing queued, nothing typed: the only moment a ghost suggestion is welcome."""
    buf = _buffer(cli)
    return (buf is not None and not buf.text and not cli._agent_running
            and cli._pending_input.empty() and cli._interrupt_queue.empty())


def _apply(cli: Any, generation: int, text: str) -> None:
    """Runs on the UI thread: the generation and idle/empty checks and the buffer write are one step,
    so a keystroke cannot land between them."""
    state = _state(cli)
    if state.generation != generation or not _idle_and_empty(cli):
        return
    state.text = text
    _buffer(cli).suggestion = Suggestion(text)
    cli._app.invalidate()


def _deliver(cli: Any, generation: int, text: Optional[str]) -> None:
    """Hand a worker-thread result to the app loop (inline when there is none, e.g. before the app runs)."""
    if not text:
        return
    loop = getattr(getattr(cli, "_app", None), "loop", None)
    if loop is not None:
        loop.call_soon_threadsafe(lambda: _apply(cli, generation, text))
    else:
        _apply(cli, generation, text)


def _request_suggestion(cli: Any) -> str:
    """Ask the session's owner for the guess (``prompt.next_prompt``); '' when there is none. The owner
    runs the call on the session's live route and bills it to the session."""
    from hermes_cli import gateway_side
    connection = gateway_side.connection_for(cli)
    if connection is None:
        return ""
    _task_id, wait = gateway_side.start(connection, cli.session_id, "next_prompt")
    return wait()


def maybe_suggest_next_prompt(cli: Any) -> None:
    """Turn-settle hook: start the background suggestion if gating allows. Never raises."""
    try:
        if (not _setting_enabled() or not _idle_and_empty(cli)
                or cli._last_turn_interrupted or cli._last_turn_failed):
            return
        state = _state(cli)
        state.text = None
        state.generation += 1
        generation = state.generation

        def work() -> None:
            try:
                _deliver(cli, generation, _request_suggestion(cli))
            except Exception as exc:
                logger.debug("Next-prompt suggestion failed: %s", exc, exc_info=True)

        _spawn(work)
    except Exception as exc:
        logger.debug("Next-prompt trigger failed: %s", exc, exc_info=True)

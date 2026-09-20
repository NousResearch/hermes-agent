"""Host-declared input origin for the current turn (never inferred from prompt text).

Context-local so queued memory work keeps the originating turn's provenance when
another session, notification or human turn starts before its worker runs.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from agent.turn_author import parse_turn_author

# Typed by the CLI/TUI process notification producers and the messaging gateway.
# `hidden` (widget/onboarding input) and `steer` are not machine-origin signals.
_RUNTIME_DISPLAY_KINDS = frozenset({
    "process_complete", "async_delegation_complete", "internal_notification",
})
_NON_USER_PLATFORMS = frozenset({"cron", "subagent"})

# Preserve the direct provider-call contract when no host turn has been bound.
_user_input: ContextVar[bool] = ContextVar("hermes_turn_user_input", default=True)


def is_user_input_turn() -> bool:
    """Whether automatic work belongs to a user-originated turn in this context."""
    return _user_input.get()


@contextmanager
def turn_input_scope(*, display_kind: str | None = None, turn_author: Any = None,
                     platform: str | None = None, staged_message: Any = None,
                     user_message: Any = None) -> Iterator[None]:
    """Bind trusted ingress metadata; restore the previous origin even on failure.

    No content matching: a human pasting a notification or a log is still a human
    input. This signal does not change execution, history, prompts or tool access;
    consumers choose their own policy (e.g. Hindsight's automatic memory hooks).
    """
    # The classic CLI types its current row before entering the agent thread.
    # Match the clean input as the turn prologue does; a stale staged row must
    # never classify an unrelated new turn.
    if (not display_kind and isinstance(staged_message, dict)
            and staged_message.get("content") == user_message):
        display_kind = staged_message.get("display_kind")
    author = parse_turn_author(turn_author)
    user_input = (
        display_kind not in _RUNTIME_DISPLAY_KINDS
        and (platform or "").lower() not in _NON_USER_PLATFORMS
        and not (author and author["is_bot"])
    )
    token = _user_input.set(user_input)
    try:
        yield
    finally:
        _user_input.reset(token)

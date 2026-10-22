"""Finalize owned browser state at every synchronous conversation exit."""
from contextlib import contextmanager
from contextvars import ContextVar


_completed: ContextVar[bool] = ContextVar('browser_turn_cleanup_completed', default=False)


def finalize_browser_for_turn(task_id: str) -> bool:
    from tools.browser_tool_lifecycle import cleanup_browser_for_turn

    confirmed = cleanup_browser_for_turn(task_id)
    if confirmed:
        _completed.set(True)
    return confirmed


@contextmanager
def browser_turn_cleanup_boundary(task_id: str):
    from tools.browser_tool_lifecycle import cleanup_browser_for_turn

    token = _completed.set(False)
    try:
        yield
    finally:
        try:
            if not _completed.get():
                cleanup_browser_for_turn(task_id)
        finally:
            _completed.reset(token)

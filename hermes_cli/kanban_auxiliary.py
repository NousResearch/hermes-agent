"""Per-card auxiliary LLM clients for kanban specify/decompose.

Each triage card gets its own client for the duration of a specify or
decompose call — mirroring how dispatched worker agents are isolated per
task — then the client is released when the operation finishes.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional, Tuple


def kanban_auxiliary_timeout(
    aux_task: str,
    client: Any,
    *,
    explicit: Optional[int] = None,
    default: float = 180.0,
) -> float:
    """Resolve the LLM call timeout for a kanban auxiliary operation."""
    from agent.auxiliary_client import _get_task_timeout

    if explicit is not None:
        timeout = float(explicit)
    else:
        timeout = _get_task_timeout(aux_task, default=default)

    try:
        from agent.cursor_auxiliary_client import (
            CursorAuxiliaryClient,
            effective_cursor_auxiliary_timeout,
        )

        if isinstance(client, CursorAuxiliaryClient):
            return effective_cursor_auxiliary_timeout(timeout)
    except ImportError:
        pass
    return timeout


@contextmanager
def kanban_card_auxiliary_client(
    task_id: str,
    aux_task: str,
) -> Iterator[Tuple[Any, Optional[str]]]:
    """Yield ``(client, model)`` scoped to one kanban task id."""
    from agent.auxiliary_client import get_isolated_text_auxiliary_client
    from agent.cursor_auxiliary_client import (
        auxiliary_operation_in_flight,
        prepare_cursor_auxiliary_credentials,
    )

    with auxiliary_operation_in_flight():
        prepare_cursor_auxiliary_credentials(reload_only=True)
        client, model = get_isolated_text_auxiliary_client(aux_task, task_id)
        try:
            yield client, model
        finally:
            release_kanban_auxiliary_client(task_id, client)


def release_kanban_auxiliary_client(task_id: str, client: Any) -> None:
    """Close a per-card auxiliary client opened for specify/decompose."""
    from agent.cursor_auxiliary_client import (
        CursorAuxiliaryClient,
        release_cursor_sdk_client,
    )

    if isinstance(client, CursorAuxiliaryClient):
        release_cursor_sdk_client(task_id)
        return
    if client is None:
        return
    # HTTP OpenAI-compatible clients: close the underlying pool.
    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass

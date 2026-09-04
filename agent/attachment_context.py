"""Turn-scoped materialized attachment context for trusted integrations.

This is the narrow runtime seam between Hermes' attachment lifecycle and
plugin tools.  A caller binds the exact attachment snapshot consumed for one
agent turn; tool worker threads inherit it through the normal ContextVar
propagation path.  The context is never a process-global "latest image" store.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


_CURRENT_ATTACHMENTS: ContextVar[tuple[dict[str, Any], ...]] = ContextVar(
    "hermes_current_turn_attachments", default=()
)


def _path_from_item(item: Any) -> str:
    if isinstance(item, Mapping):
        value = item.get("path", item.get("materialized_path"))
    else:
        value = item
    return str(value) if value is not None else ""


def _metadata_from_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {}
    value = item.get("metadata", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _reference_from_item(item: Any) -> str | None:
    if not isinstance(item, Mapping):
        return None
    value = item.get("reference", item.get("id"))
    return str(value) if value is not None and str(value).strip() else None


def _snapshot(
    attachments: Iterable[Any] | None,
    *,
    session_id: str,
    task_id: str,
    turn_id: str,
    surface: str,
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for index, item in enumerate(attachments or ()):
        path = _path_from_item(item)
        reference = _reference_from_item(item) or (
            f"hermes:{session_id}:{turn_id}:attachment:{index}"
        )
        metadata = _metadata_from_item(item)
        metadata.update(
            {
                "surface": surface or "unknown",
                "session_id": session_id,
                "task_id": task_id,
                "turn_id": turn_id,
                "attachment_index": index,
                "materialized_by": "hermes",
                "filename": Path(path).name if path else None,
            }
        )
        result.append(
            {
                "path": path,
                "reference": reference,
                "kind": "incoming_attachment",
                "metadata": metadata,
            }
        )
    return tuple(result)


@contextmanager
def bind_current_turn_attachments(
    attachments: Iterable[Any] | None,
    *,
    session_id: str,
    task_id: str,
    turn_id: str,
    surface: str = "",
) -> Iterator[None]:
    """Bind one immutable snapshot for the duration of an agent turn.

    The caller must pass the list it already owns for this turn.  This helper
    does not search Hermes storage, infer a file from a directory, or delete
    the attachment.  The returned context is reset even when execution fails.
    """
    token: Token[tuple[dict[str, Any], ...]] = _CURRENT_ATTACHMENTS.set(
        _snapshot(
            attachments,
            session_id=str(session_id or ""),
            task_id=str(task_id or ""),
            turn_id=str(turn_id or ""),
            surface=str(surface or ""),
        )
    )
    try:
        yield
    finally:
        _CURRENT_ATTACHMENTS.reset(token)


def get_current_turn_attachments(
    *,
    session_id: str | None = None,
    task_id: str | None = None,
    turn_id: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return a defensive copy of the exact active turn attachment snapshot.

    Optional identity checks make accidental cross-session/tool reuse fail
    closed.  An empty tuple means no current-turn attachment is bound.
    """
    current = _CURRENT_ATTACHMENTS.get()
    if session_id is not None and current:
        actual = current[0].get("metadata", {}).get("session_id")
        if str(session_id or "") != str(actual or ""):
            return ()
    if task_id is not None and current:
        actual = current[0].get("metadata", {}).get("task_id")
        if str(task_id or "") != str(actual or ""):
            return ()
    if turn_id is not None and current:
        actual = current[0].get("metadata", {}).get("turn_id")
        if str(turn_id or "") != str(actual or ""):
            return ()
    return tuple(
        {
            **attachment,
            "metadata": dict(attachment.get("metadata", {})),
        }
        for attachment in current
    )


__all__ = ["bind_current_turn_attachments", "get_current_turn_attachments"]

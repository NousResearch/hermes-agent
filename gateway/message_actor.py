"""Separate the authenticated message actor from its conversation route."""
import copy
import dataclasses
from typing import Any, Optional

def _identity_string(value: Any) -> Optional[str]:
    """Return a concrete actor identifier, ignoring dynamic mock attributes."""
    if isinstance(value, str):
        return value or None
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def event_actor_identity(event: Any) -> tuple[Optional[str], Optional[str]]:
    """Resolve a real actor from a MessageEvent-like object.

    Missing attributes on lightweight mocks can be synthesized as mock objects;
    those values are not identities and must fall back to the routing source.
    """
    source = getattr(event, "source", None)
    event_user_id = _identity_string(getattr(event, "user_id", None))
    source_user_id = _identity_string(getattr(source, "user_id", None))
    event_user_name = getattr(event, "user_name", None)
    source_user_name = getattr(source, "user_name", None)
    return (
        event_user_id or source_user_id,
        event_user_name
        if isinstance(event_user_name, str) and event_user_name
        else source_user_name
        if isinstance(source_user_name, str) and source_user_name
        else None,
    )


def copy_session_source_with(source: Any, **changes: Any) -> Any:
    """Shallow-copy a SessionSource-like object and apply access-only fields.

    A shallow copy supports real SessionSource instances and lightweight plugin
    or test objects while preserving dynamic transport/profile provenance.
    Failure returns the original source, keeping authorization fail-closed.
    """
    if source is None:
        return None
    try:
        if all(getattr(source, name, None) == value for name, value in changes.items()):
            return source
    except Exception:
        pass
    try:
        copied = copy.copy(source)
        if copied is source:
            raise TypeError("source copy returned the original object")
        for name, value in changes.items():
            setattr(copied, name, value)
        return copied
    except Exception:
        try:
            if dataclasses.is_dataclass(source) and not isinstance(source, type):
                return dataclasses.replace(source, **changes)
        except Exception:
            pass
    return source


def source_for_event_actor(event: Any) -> Any:
    """Return an access-only source carrying the event's resolved real actor."""
    source = getattr(event, "source", None)
    user_id, user_name = event_actor_identity(event)
    return copy_session_source_with(source, user_id=user_id, user_name=user_name)

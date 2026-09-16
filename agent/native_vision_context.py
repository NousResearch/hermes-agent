"""Turn-local visibility for images already attached to the main model."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterable, Iterator


_NATIVE_IMAGE_REFS: ContextVar[frozenset[str]] = ContextVar(
    "hermes_native_image_refs",
    default=frozenset(),
)


def _normalize_ref(ref: str) -> str:
    value = str(ref or "").strip()
    if not value:
        return ""
    if value.startswith("file://"):
        value = value[7:]
    if value.startswith(("http://", "https://", "data:")):
        return value
    try:
        return str(Path(value).expanduser().resolve(strict=False))
    except Exception:
        return value


@contextmanager
def scoped_native_image_refs(refs: Iterable[str]) -> Iterator[None]:
    """Publish image references visible in the current parent agent turn."""
    normalized = frozenset(filter(None, (_normalize_ref(ref) for ref in refs)))
    token = _NATIVE_IMAGE_REFS.set(normalized)
    try:
        yield
    finally:
        _NATIVE_IMAGE_REFS.reset(token)


def is_native_image_attached(ref: str) -> bool:
    """Return whether *ref* is already visible to the current main model."""
    # Children inherit ContextVars, but their user turn does not contain the
    # parent's image parts. They must remain able to load the referenced image.
    from agent.delegation_context import is_delegated_child_context

    if is_delegated_child_context():
        return False
    normalized = _normalize_ref(ref)
    return bool(normalized and normalized in _NATIVE_IMAGE_REFS.get())

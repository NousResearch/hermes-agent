"""NOVA error types.

Configuration errors carry the file and the dotted field path that produced them,
because the reader is usually a customer's engineer looking at their own YAML rather
than a NOVA developer looking at a traceback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class NovaError(Exception):
    """Base class for every error this package raises deliberately."""


class SpecError(NovaError):
    """A tenant bundle is malformed, incomplete, or internally inconsistent.

    ``field`` is a dotted path into the document (``model.provider``,
    ``limits.max_retries``); ``source`` is the file it came from. Both are optional
    because some checks are cross-file and belong to the bundle rather than one
    document.
    """

    def __init__(
        self,
        message: str,
        *,
        field: Optional[str] = None,
        source: Optional[Path] = None,
    ) -> None:
        self.message = message
        self.field = field
        self.source = source
        super().__init__(self._render())

    def _render(self) -> str:
        where = []
        if self.source is not None:
            where.append(str(self.source))
        if self.field:
            where.append(self.field)
        return f"{': '.join(where)}: {self.message}" if where else self.message


class RuntimeAdapterError(NovaError):
    """A runtime adapter could not carry out a request.

    Raised for conditions the adapter detects about the runtime itself — a missing
    home directory, a refusal to overwrite state NOVA does not own — never for
    malformed specs, which are :class:`SpecError`.
    """


class AuditError(NovaError):
    """The audit log could not record an event.

    Treated as fatal by callers: under "model-visible means logged", an unlogged
    change to model-visible state must not be allowed to happen.
    """

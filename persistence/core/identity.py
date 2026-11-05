"""Identity, paging, and value-shape primitives shared by persistence capabilities.

Every identifier here is an opaque stable anchor. Backends mint or accept them, order them internally however they like, and callers never assume dense rowids, numeric shape, or global monotonicity. Capability modules import these shared types and define capability-specific identifier aliases.

A ``Page`` carries an opaque ``ContinuationToken`` or ``None`` at the end of the result set. Raw cursors and live iterators over wire rows are not exposed at this boundary; ``TranscriptPageRequest`` separately supports offset-based transcript reads. The cursor guarantee is no loss and no duplication over an unchanged dataset; snapshot stability is not promised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, List, NewType, Optional, TypeAlias, TypeVar, Union

__all__ = [
    "ClaimToken",
    "ContinuationToken",
    "DelegationId",
    "JsonValue",
    "MessageId",
    "Page",
    "PageRequest",
    "SessionId",
]

# Opaque stable anchors. SessionId is caller-supplied today; DelegationId and ClaimToken are backend-minted strings. MessageId is a backend-minted integer that readback surfaces as m["id"] and that rewind/search/pagination reuse as a stable anchor. Opacity here means no density, arithmetic, or global-monotonicity assumptions; ordering promises are always (timestamp, id) with id as the deterministic tiebreaker.
SessionId = NewType("SessionId", str)
MessageId = NewType("MessageId", int)
DelegationId = NewType("DelegationId", str)
ClaimToken = NewType("ClaimToken", str)

# Opaque keyset cursor. Only ever produced by a backend and handed
# back verbatim; callers never parse or construct one from parts.
ContinuationToken = NewType("ContinuationToken", str)


# Backend-neutral JSON payload shape. Recursive on purpose: message content,
# metadata, and routing payloads are trees, never backend-serialized blobs.
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PageRequest:
    """One keyset page ask: a positive ``limit`` and the cursor from the previous page, if any."""

    limit: int
    cursor: Optional[ContinuationToken] = None

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError(f"PageRequest.limit must be >= 1, got {self.limit}")


@dataclass(frozen=True, slots=True)
class Page(Generic[T]):
    """One keyset page of results; ``continuation is None`` means the result set is exhausted."""

    items: tuple[T, ...]
    continuation: Optional[ContinuationToken]

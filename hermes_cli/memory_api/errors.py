"""Phase 4 — Memory API errors.

Errors are explicit and typed. The API never reports a "successful" write that
was not persisted: an unsupported or unavailable write raises rather than
returning a silent no-op success (docs/memory/memory-architecture.md §16.4).
"""

from __future__ import annotations

from typing import Optional


class MemoryAPIError(Exception):
    """Base class for all Memory API errors."""


class CapabilityError(MemoryAPIError):
    """A requested memory operation could not be performed by any provider.

    Raised (never swallowed into a fake success) when a write/query operation
    has no registered, available capability that can fulfill it, or when the
    capability explicitly refuses (e.g. a read-only backend asked to write).
    """

    def __init__(
        self,
        operation: str,
        reason: str,
        *,
        layer: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        self.operation = operation
        self.reason = reason
        self.layer = layer
        self.provider = provider
        bits = [f"operation={operation!r}"]
        if layer:
            bits.append(f"layer={layer}")
        if provider:
            bits.append(f"provider={provider}")
        bits.append(f"reason={reason}")
        super().__init__("capability unavailable: " + ", ".join(bits))


class UnsupportedCapability(CapabilityError):
    """A specific capability/provider is not registered or not available.

    Distinguishes "not built yet" (L2/L4 writers in later phases) from a
    transient outage. Callers can branch on this type to degrade gracefully
    without treating it as a generic failure.
    """

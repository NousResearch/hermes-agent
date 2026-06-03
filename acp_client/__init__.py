"""Hermes-as-ACP-client subsystem (Phase 1 skeleton).

Hermes already ships an ACP **server** (``acp_adapter/``) so editors such as
Zed can drive Hermes as an agent.  This package is the symmetric, opt-in
**client** side: it lets Hermes drive an *external* ACP agent process
(``claude``, ``codex``, ``gemini-cli``, a sibling ``hermes acp`` …) over the
same ``acp`` PyPI library that already backs the server.

Phase 1 is a **library-only skeleton**: no CLI server entry, no wiring into the
Kanban worker or ``delegate_task``, no real external CLI launch.  The public
import surface is intentionally tiny::

    from acp_client.connection import OutboundConnection

Design + provenance:
``spearhead-execution/20260529-acpx-interop-spike/acp-acpx-interop-design.md``
(§2 architecture, §4 phased plan).  acpx (OpenClaw) is shape inspiration only —
nothing is vendored and no new third-party dependency is introduced; the
existing optional ``hermes-agent[acp]`` extra is the only requirement.
"""

from __future__ import annotations

# Re-exported lazily-importable names.  Importing this package must not require
# the optional ``acp`` dependency, mirroring ``acp_adapter``'s tolerance for a
# missing extra (see ``acp_adapter/entry.py``).
__all__ = [
    "OutboundConnection",
    "OutboundSessionManager",
    "OutboundSessionState",
    "PermissionRelay",
    "EventTranslator",
    "TransportRegistry",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy import shim
    if name == "OutboundConnection":
        from acp_client.connection import OutboundConnection

        return OutboundConnection
    if name in {"OutboundSessionManager", "OutboundSessionState"}:
        from acp_client import outbound_session as _mod

        return getattr(_mod, name)
    if name == "PermissionRelay":
        from acp_client.permission_relay import PermissionRelay

        return PermissionRelay
    if name == "EventTranslator":
        from acp_client.event_translator import EventTranslator

        return EventTranslator
    if name == "TransportRegistry":
        from acp_client.transport_registry import TransportRegistry

        return TransportRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

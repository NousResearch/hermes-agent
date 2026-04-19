"""Identity Layer — Hermes input-pipeline submodule C§1.1.

Consumes a SessionBootstrap and produces an IdentityPacket that binds
every downstream submodule to a verified (user_id, company_id, mode, time)
tuple for the lifetime of the request.

Phase-3 build plan reference: §C§1 table, row 1.
Wire-up to the central Hermes entrypoint is task C§1.9 (not this file).

Event emitted: ``hermes.identity.bootstrap``
Emission mechanism: stdout JSON line (single-line, newline-terminated).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# I/O types
# ---------------------------------------------------------------------------


class SessionBootstrap(BaseModel):
    """Minimal envelope delivered at session start before identity is resolved."""

    raw_user_id: Optional[str] = None
    raw_company_id: Optional[str] = None
    session_id: str
    source: str  # e.g. "telegram", "slack", "cli"
    metadata: dict = Field(default_factory=dict)


class IdentityPacket(BaseModel):
    """Resolved, authoritative identity for one Hermes request turn."""

    user_id: str
    company_id: Optional[str]
    mode: Literal["personal", "enterprise", "anonymous"]
    time: datetime


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


def _emit(event: str, payload: dict) -> None:
    """Write a single-line JSON event to stdout.

    Replace with an @agrv/hermes-events call in C§1.9 when the shared
    event bus is wired into this workspace.
    """
    line = json.dumps({"event": event, **payload}, default=str)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Submodule entry point
# ---------------------------------------------------------------------------


def bootstrap_identity(bootstrap: SessionBootstrap) -> IdentityPacket:
    """Resolve a SessionBootstrap into a verified IdentityPacket.

    Stub implementation: passes raw IDs through and infers mode from
    the presence of a company_id. Full implementation (auth token
    validation, DB look-up via memory-engine) is deferred to C§1.9.

    Emits ``hermes.identity.bootstrap`` on success.
    """
    user_id = bootstrap.raw_user_id or f"anon:{bootstrap.session_id}"
    company_id = bootstrap.raw_company_id or None

    if bootstrap.raw_user_id is None:
        mode: Literal["personal", "enterprise", "anonymous"] = "anonymous"
    elif company_id:
        mode = "enterprise"
    else:
        mode = "personal"

    packet = IdentityPacket(
        user_id=user_id,
        company_id=company_id,
        mode=mode,
        time=datetime.now(tz=timezone.utc),
    )

    _emit(
        "hermes.identity.bootstrap",
        {
            "user_id": packet.user_id,
            "company_id": packet.company_id,
            "mode": packet.mode,
            "session_id": bootstrap.session_id,
        },
    )

    return packet

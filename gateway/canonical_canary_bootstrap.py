"""Private, one-shot bootstrap types for an isolated Canonical canary.

This module is deliberately outside the Canonical Writer wire protocol.  Only
the root-configured writer bootstrap may construct this request.  Its database
authority is separately provisioned and consumed in one transaction.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class CanaryScopeBootstrapRequest:
    grant_id: str
    case_id: str
    release_sha256: str
    fixture_sha256: str
    run_id: str
    session_key_sha256: str
    expires_at: dt.datetime
    approved_by: str
    approval_source_sha256: str
    provisioning_receipt_sha256: str


@dataclass(frozen=True)
class CanaryScopePreclaimRetirementRequest:
    """Exact sealed scope used by writer-only preclaim reconciliation."""

    grant_id: str
    case_id: str
    release_sha256: str
    fixture_sha256: str
    run_id: str
    session_key_sha256: str
    expires_at: dt.datetime
    approved_by: str
    approval_source_sha256: str
    provisioning_receipt_sha256: str


__all__ = [
    "CanaryScopeBootstrapRequest",
    "CanaryScopePreclaimRetirementRequest",
]

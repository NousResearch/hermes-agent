"""The coding-workspace claim rule the store applies when it reserves a local session's creation receipt.

A Desktop coding chat freezes its prepared checkout (``CODING_WORKSPACE_KEY`` in the policy's ``config_json``),
keyed by the client's workspace request id. Two receipts for different sessions may never name the same
request: the second principal's creation is refused inside the same write that would store it.

Standard library only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

#: The frozen policy's ``config_json`` key holding the prepared coding workspace.
CODING_WORKSPACE_KEY = "canonical_coding_workspace_v1"


def workspace_request_id(policy: Mapping[str, Any]) -> str | None:
    """The workspace request a receipt's policy claims, or None when it is no coding chat. Raises
    ``ValueError`` / ``TypeError`` / ``KeyError`` for a malformed policy."""
    prepared = json.loads(policy["config_json"]).get(CODING_WORKSPACE_KEY)
    return prepared["binding"]["requestId"] if prepared else None


def claim_conflict(receipt: Mapping[str, Any], stored: Iterable[Any]) -> bool:
    """Whether a stored receipt (each a decoded dict or its JSON text) already owns *receipt*'s workspace
    request for another session. A receipt that claims no workspace (or none readable) conflicts with
    nothing; malformed unrelated receipts name no owner (as in retirement scans)."""
    try:
        request = workspace_request_id(receipt["policy"])
    except (ValueError, TypeError, KeyError, AttributeError):
        return False
    if request is None:
        return False
    for value in stored:
        try:
            other = json.loads(value) if isinstance(value, (str, bytes)) else value
            if workspace_request_id(other["policy"]) == request and other["session_id"] != receipt["session_id"]:
                return True
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
    return False

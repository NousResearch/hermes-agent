"""The authority family's shared vocabulary: the admission digest and the constants admissions, workers and the gateway agree on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ADMISSION_QUEUE_MAX",
    "ADMITTED_REQUEST_CONFLICT",
    "CRON_PRINCIPAL_ID",
    "CRON_REQUEST_PREFIX",
    "LEGACY_CREATION_SOURCES",
    "LEGACY_LOCAL_SOURCES",
    "NOT_CLIENT_LABELS",
    "RETRY_HORIZON_S",
    "STEERING",
    "OwnerRule",
    "admission_fingerprint",
]

#: Queued inputs one session holds; ``admit`` refuses the next with ``queue_full``.
ADMISSION_QUEUE_MAX = 32

#: The field and detail of the one ``admission_conflict`` that proves the request was already admitted: this principal's request id on this target holds an admission with another payload. Every other conflict (route, binding, lineage) carries no such proof, so a client tells them apart by this field.
ADMITTED_REQUEST_CONFLICT = {"field": "request_id", "detail": "this request id was already admitted with another payload"}

#: The outcome of a queued admission handed to its FIFO's running claim, bound to that claim's ``(owner_epoch, generation)`` until it settles ``steered``, ``interrupted`` or back to ``queued``.
STEERING = "steering"

#: The scheduler's principal and the prefix of its request ids (``cron:<job>:<execution>``).
CRON_PRINCIPAL_ID = "cron-owner"
CRON_REQUEST_PREFIX = "cron:"

#: Local surfaces whose unrouted rows are owner history the native resume adopts.
LEGACY_LOCAL_SOURCES = frozenset({"cli", "tui", "gui", "desktop", "oneshot", "tool", "acp"})

#: Legacy labels that are not creation sources of their own.
LEGACY_CREATION_SOURCES = {"desktop": "gui"}

#: Labels a routeless row can carry that never name a chat app: automation and recovery rows, a retired platform, and a third-party WebUI that runs its own agent on this store (its rows stay read-only).
NOT_CLIENT_LABELS = frozenset({"cron", "subagent", "kanban", "bot_room", "a2a", "webhook", "api_server", "import", "recovered",
                               "curator", "batch", "homeassistant", "webui", "local", "unknown"})


@dataclass(frozen=True, slots=True)
class OwnerRule:
    """The host data the owner rule reads (``session_rules.legacy_local.same_owner``): the host owner's own OS subject (``uid:<n>`` / ``sid:<sid>``), the dashboard sign-in providers whose stamps name that owner, and the labels of every messaging platform Hermes ships (a routeless row under one is no client app's chat). The caller builds it; operations that compare a stored owner take it as data."""

    host_subject: str
    login_providers: frozenset[str]
    messaging_labels: frozenset[str]

#: Exact-retry horizon: a retry within it gets the byte-exact settled reply; sweep tombstones age out after it.
RETRY_HORIZON_S = 7 * 86400


def admission_fingerprint(*, canonical_target: str, payload: Any) -> str:
    """Canonical payload identity after authorization and immutable media capture."""
    encoded = json.dumps(
        {"target": canonical_target, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(encoded).hexdigest()

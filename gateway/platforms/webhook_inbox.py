"""Durable per-delivery records for the webhook adapter (restart safety).

One small JSON file per accepted delivery under ``<HERMES_HOME>/state/webhook_inbox/``:

* ``pending``    -- accepted (the sender got 2xx) but not yet handed to the runner. Carries the
                    replayable work (payload, rendered prompt, event type) and is written with
                    fsync BEFORE the 2xx, so a crash at any later point replays it on restart.
* ``started``    -- handed to the runner. The work is dropped; the record is only the
                    idempotency tombstone plus the reply envelope ``send()`` needs.
* ``superseded`` -- folded into a newer coalesced event; tombstone only.

Each transition rewrites only that delivery's file, so a delivery costs O(1) bytes no matter
how many deliveries are live inside the idempotency TTL (a whole-state snapshot per delivery
was O(n) per write, O(n^2) per TTL window). Writes are synchronous here; the adapter runs
them on a worker thread so ``fsync`` never lands on the event loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from utils import atomic_json_write

logger = logging.getLogger(__name__)

INBOX_DIRNAME = "webhook_inbox"
PENDING = "pending"
STARTED = "started"
SUPERSEDED = "superseded"
_STATES = frozenset({PENDING, STARTED, SUPERSEDED})
_SCHEMA = 1


def record_filename(profile: str, route: str, delivery_id: str) -> str:
    """Stable, filesystem-safe name; provider-controlled ids never become path components."""
    digest = hashlib.sha256(
        json.dumps((profile, route, delivery_id), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{digest}.json"


class WebhookInbox:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path_for(self, profile: str, route: str, delivery_id: str) -> Path:
        return self.root / record_filename(profile, route, delivery_id)

    def write(self, record: Dict[str, Any]) -> None:
        """Atomically replace one delivery's record (temp file + fsync + rename, 0600: payloads
        can carry provider data)."""
        self.root.mkdir(parents=True, exist_ok=True)
        data = {**record, "schema": _SCHEMA}
        atomic_json_write(
            self.path_for(record["profile"], record["route"], record["delivery_id"]),
            data, indent=None, mode=0o600, default=str,
        )

    def delete(self, profile: str, route: str, delivery_id: str) -> None:
        try:
            self.path_for(profile, route, delivery_id).unlink()
        except FileNotFoundError:
            pass

    def load(self) -> List[Dict[str, Any]]:
        """Every well-formed record, oldest first. Malformed files are skipped (and left for an
        operator) rather than failing adapter start-up."""
        try:
            names = sorted(os.listdir(self.root))
        except (FileNotFoundError, NotADirectoryError):
            return []
        records = []
        for name in names:
            if not name.endswith(".json") or name.startswith("."):
                continue
            try:
                record = json.loads((self.root / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logger.warning("[webhook] skipping unreadable inbox record %s", name)
                continue
            if not (
                isinstance(record, dict)
                and record.get("state") in _STATES
                and all(isinstance(record.get(key), str) for key in ("profile", "route", "delivery_id"))
                and isinstance(record.get("at"), (int, float))
                and name == record_filename(record["profile"], record["route"], record["delivery_id"])
            ):
                logger.warning("[webhook] skipping malformed inbox record %s", name)
                continue
            if record["state"] == PENDING and not isinstance(record.get("work"), dict):
                logger.warning("[webhook] skipping pending inbox record %s without its work", name)
                continue
            records.append(record)
        records.sort(key=lambda r: float(r["at"]))
        return records

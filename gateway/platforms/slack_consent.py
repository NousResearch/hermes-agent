"""Persistent consent state for the Slack channel consent gate.

When ``channel_consent_gate`` is enabled in the Slack platform config, the
bot stays dormant in newly joined channels until a human explicitly
activates it via Block Kit buttons. This module owns the on-disk state:
a small JSON map of channel_id → consent record.

States:
  (absent)   — channel predates the gate or gate never saw the join;
               bot behaves normally (the gate only governs channels it
               watched the bot join).
  pending    — bot joined, consent prompt posted, no decision yet → dormant.
  approved   — a human clicked Activate → bot behaves normally.
  declined   — a human clicked Decline → dormant.

The file lives at ``$HERMES_HOME/slack_channel_consent.json``. Writes are
atomic (tmp + rename). All operations are defensive: a corrupt or
unreadable file degrades to an empty store rather than raising into the
event pipeline.
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

PENDING = "pending"
APPROVED = "approved"
DECLINED = "declined"

_VALID_STATES = {PENDING, APPROVED, DECLINED}


class ChannelConsentStore:
    """JSON-backed map of Slack channel_id → consent record."""

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            from hermes_constants import get_hermes_home

            path = get_hermes_home() / "slack_channel_consent.json"
        self._path = Path(path)
        self._data: Dict[str, dict] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data = {
                        str(k): v
                        for k, v in raw.items()
                        if isinstance(v, dict)
                        and v.get("status") in _VALID_STATES
                    }
        except Exception:
            logger.warning(
                "[Slack] Could not read channel consent store at %s; "
                "starting empty",
                self._path,
                exc_info=True,
            )
            self._data = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=str(self._path.parent), prefix=".consent-", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=2, sort_keys=True)
                os.replace(tmp, self._path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except Exception:
            logger.warning(
                "[Slack] Could not persist channel consent store at %s",
                self._path,
                exc_info=True,
            )

    def status(self, channel_id: str) -> Optional[str]:
        """Return 'pending' | 'approved' | 'declined', or None if untracked."""
        self._load()
        rec = self._data.get(channel_id)
        return rec.get("status") if rec else None

    def is_dormant(self, channel_id: str) -> bool:
        """True when the bot must not process messages in this channel."""
        return self.status(channel_id) in {PENDING, DECLINED}

    def set(
        self,
        channel_id: str,
        status: str,
        *,
        by_user_id: str = "",
        by_user_name: str = "",
    ) -> None:
        if status not in _VALID_STATES:
            raise ValueError(f"invalid consent status: {status!r}")
        self._load()
        self._data[str(channel_id)] = {
            "status": status,
            "by_user_id": by_user_id,
            "by_user_name": by_user_name,
            "updated_at": int(time.time()),
        }
        self._save()

    def forget(self, channel_id: str) -> None:
        """Drop a channel from the store (reverts to untracked/normal)."""
        self._load()
        if self._data.pop(str(channel_id), None) is not None:
            self._save()

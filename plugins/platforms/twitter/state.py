from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)
STATE_VERSION = 1


def state_path() -> Path:
    return get_hermes_home() / "twitter" / "state.json"


@dataclass
class TwitterState:
    mention_since_id: str = ""
    dm_since_id: str = ""
    _seen: OrderedDict[str, None] = field(default_factory=OrderedDict)
    _branches: OrderedDict[str, str] = field(default_factory=OrderedDict)
    max_seen: int = 2000
    max_branches: int = 2000

    @classmethod
    def load(
        cls, *, max_seen: int = 2000, max_branches: int = 2000
    ) -> "TwitterState":
        path = state_path()
        try:
            payload = json.loads(path.read_text())
            if payload.get("version") != STATE_VERSION:
                raise ValueError("unsupported state version")
            state = cls(
                mention_since_id=str(payload.get("mention_since_id") or ""),
                dm_since_id=str(payload.get("dm_since_id") or ""),
                max_seen=max_seen,
                max_branches=max_branches,
            )
            for item in payload.get("seen_ids") or []:
                state._seen[str(item)] = None
            for item in payload.get("bot_post_anchors") or []:
                if isinstance(item, list) and len(item) == 2:
                    state._branches[str(item[0])] = str(item[1])
            state._trim()
            return state
        except FileNotFoundError:
            return cls(max_seen=max_seen, max_branches=max_branches)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            cls._quarantine(path)
            return cls(max_seen=max_seen, max_branches=max_branches)

    @staticmethod
    def _quarantine(path: Path) -> None:
        if not path.exists():
            return
        destination = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
        try:
            path.replace(destination)
            logger.warning("Quarantined corrupt Twitter state at %s", destination)
        except OSError:
            logger.warning("Ignoring unreadable Twitter state at %s", path)

    def save(self) -> None:
        payload: dict[str, Any] = {
            "version": STATE_VERSION,
            "mention_since_id": self.mention_since_id,
            "dm_since_id": self.dm_since_id,
            "seen_ids": list(self._seen),
            "bot_post_anchors": [list(item) for item in self._branches.items()],
        }
        atomic_json_write(state_path(), payload)

    def seen(self, event_id: str) -> bool:
        return str(event_id) in self._seen

    def mark_seen(self, event_id: str) -> None:
        key = str(event_id)
        self._seen.pop(key, None)
        self._seen[key] = None
        self._trim()

    def map_bot_post(self, post_id: str, anchor_id: str) -> None:
        key = str(post_id)
        self._branches.pop(key, None)
        self._branches[key] = str(anchor_id)
        self._trim()

    def resolve_anchor(self, trigger_id: str, ancestor_ids: list[str]) -> str:
        for ancestor_id in ancestor_ids:
            anchor = self._branches.get(str(ancestor_id))
            if anchor:
                return anchor
        return str(trigger_id)

    def advance_mentions(self, post_id: str) -> None:
        self.mention_since_id = str(post_id)

    def advance_dms(self, event_id: str) -> None:
        self.dm_since_id = str(event_id)

    def _trim(self) -> None:
        while len(self._seen) > self.max_seen:
            self._seen.popitem(last=False)
        while len(self._branches) > self.max_branches:
            self._branches.popitem(last=False)

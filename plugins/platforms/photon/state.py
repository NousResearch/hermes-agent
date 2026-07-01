"""Persistent Photon correlation state.

This store intentionally keeps only bounded delivery/reaction metadata.  It is
not a message archive and should never hold message bodies, attachment bytes, or
Photon credentials.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_SENT_MAX = 1000
DEFAULT_LAST_INBOUND_MAX = 200
DEFAULT_REACTIONS_MAX = 512
DEFAULT_AUDIT_MAX = 500
DEFAULT_RETENTION_SECONDS = 48 * 3600


def photon_state_path() -> Path:
    return get_hermes_home() / "plugins" / "photon" / "state.json"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_ts(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _short_error(error: Any, *, limit: int = 300) -> Optional[str]:
    if error is None:
        return None
    text = str(error)
    return text[:limit]


class PhotonStateStore:
    """Small plugin-local JSON store for Photon send/reaction correlation."""

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        sent_max: int = DEFAULT_SENT_MAX,
        last_inbound_max: int = DEFAULT_LAST_INBOUND_MAX,
        reactions_max: int = DEFAULT_REACTIONS_MAX,
        audit_max: int = DEFAULT_AUDIT_MAX,
        retention_seconds: int = DEFAULT_RETENTION_SECONDS,
    ) -> None:
        self.path = path or photon_state_path()
        self.sent_max = sent_max
        self.last_inbound_max = last_inbound_max
        self.reactions_max = reactions_max
        self.audit_max = audit_max
        self.retention_seconds = retention_seconds
        self.load_error: Optional[str] = None
        self.write_error: Optional[str] = None
        self._state = self._empty_state()

    @staticmethod
    def _empty_state() -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "updated_at": None,
            "sent_messages": {},
            "last_inbound_by_chat": {},
            "reactions": {},
            "audit": [],
        }

    def load(self) -> Dict[str, Any]:
        self.load_error = None
        if not self.path.exists():
            self._state = self._empty_state()
            return self.snapshot()
        try:
            raw = self.path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("state root is not an object")
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("unsupported schema version")
            self._state = self._normalize(payload)
        except Exception as exc:
            self.load_error = str(exc)
            logger.warning(
                "[photon] ignoring unreadable persistent state at %s: %s",
                self.path,
                exc,
            )
            self._state = self._empty_state()
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "schema_version": self._state["schema_version"],
            "updated_at": self._state.get("updated_at"),
            "sent_messages": dict(self._state["sent_messages"]),
            "last_inbound_by_chat": dict(self._state["last_inbound_by_chat"]),
            "reactions": dict(self._state["reactions"]),
            "audit": list(self._state["audit"]),
            "path": str(self.path),
            "load_error": self.load_error,
            "write_error": self.write_error,
        }

    def health(self) -> Dict[str, Any]:
        state = self.snapshot()
        reactions = state["reactions"]
        failures = [
            item for item in state["audit"]
            if isinstance(item, dict) and item.get("status") == "failed"
        ]
        return {
            "path": state["path"],
            "schema_version": state["schema_version"],
            "load_error": state["load_error"],
            "write_error": state["write_error"],
            "sent_messages": len(state["sent_messages"]),
            "last_inbound_chats": len(state["last_inbound_by_chat"]),
            "active_reactions": sum(
                1 for item in reactions.values()
                if isinstance(item, dict) and not item.get("removed_at")
            ),
            "audit_entries": len(state["audit"]),
            "last_failure": failures[-1] if failures else None,
        }

    def record_sent_message(
        self,
        message_id: Optional[str],
        *,
        chat_key: Optional[str] = None,
        space_id: Optional[str] = None,
        kind: str = "text",
    ) -> None:
        if not message_id:
            return
        self._state["sent_messages"][str(message_id)] = {
            "chat_key": chat_key,
            "space_id": space_id,
            "sent_at": _now_iso(),
            "kind": kind,
        }
        self._persist()

    def record_last_inbound(
        self,
        chat_key: Optional[str],
        message_id: Optional[str],
        *,
        space_id: Optional[str] = None,
    ) -> None:
        if not chat_key or not message_id:
            return
        self._state["last_inbound_by_chat"][str(chat_key)] = {
            "message_id": str(message_id),
            "space_id": space_id,
            "seen_at": _now_iso(),
        }
        self._persist()

    def record_reaction_added(
        self,
        space_id: Optional[str],
        message_id: Optional[str],
        emoji: str,
        reaction_id: Optional[str],
    ) -> None:
        if not space_id or not message_id:
            return
        key = self.reaction_key(space_id, message_id)
        self._state["reactions"][key] = {
            "reaction_id": reaction_id,
            "emoji": emoji,
            "created_at": _now_iso(),
            "removed_at": None,
        }
        self._persist()

    def record_reaction_removed(
        self,
        space_id: Optional[str],
        message_id: Optional[str],
        *,
        succeeded: bool,
    ) -> None:
        if not space_id or not message_id:
            return
        key = self.reaction_key(space_id, message_id)
        slot = self._state["reactions"].get(key)
        if not isinstance(slot, dict):
            return
        if succeeded:
            slot["removed_at"] = _now_iso()
        self._persist()

    def reaction_for(
        self, space_id: Optional[str], message_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not space_id or not message_id:
            return None
        slot = self._state["reactions"].get(self.reaction_key(space_id, message_id))
        if not isinstance(slot, dict) or slot.get("removed_at"):
            return None
        return dict(slot)

    def record_audit(
        self,
        *,
        action: str,
        status: str,
        chat_key: Optional[str] = None,
        message_id: Optional[str] = None,
        reaction_id: Optional[str] = None,
        error_class: Optional[str] = None,
        error: Any = None,
    ) -> None:
        self._state["audit"].append({
            "at": _now_iso(),
            "action": action,
            "status": status,
            "chat_key": chat_key,
            "message_id": message_id,
            "reaction_id": reaction_id,
            "error_class": error_class,
            "error": _short_error(error),
        })
        self._persist()

    @staticmethod
    def reaction_key(space_id: str, message_id: str) -> str:
        return f"{space_id}\0{message_id}"

    def _persist(self) -> None:
        self.write_error = None
        self._state = self._normalize(self._state)
        self._state["updated_at"] = _now_iso()
        try:
            atomic_json_write(
                self.path,
                self._state,
                indent=2,
                mode=0o600,
                sort_keys=True,
            )
        except Exception as exc:
            self.write_error = str(exc)
            logger.warning("[photon] failed to persist state at %s: %s", self.path, exc)

    def _normalize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = time.time()
        cutoff = now - self.retention_seconds
        state = self._empty_state()
        state["updated_at"] = payload.get("updated_at")

        sent = payload.get("sent_messages")
        if isinstance(sent, dict):
            for message_id, value in sent.items():
                if not isinstance(message_id, str) or not isinstance(value, dict):
                    continue
                ts = _parse_ts(value.get("sent_at"))
                if ts and ts < cutoff:
                    continue
                state["sent_messages"][message_id] = {
                    "chat_key": _string_or_none(value.get("chat_key")),
                    "space_id": _string_or_none(value.get("space_id")),
                    "sent_at": value.get("sent_at") if isinstance(value.get("sent_at"), str) else _now_iso(),
                    "kind": _string_or_none(value.get("kind")) or "text",
                }

        inbound = payload.get("last_inbound_by_chat")
        if isinstance(inbound, dict):
            for chat_key, value in inbound.items():
                if not isinstance(chat_key, str) or not isinstance(value, dict):
                    continue
                message_id = _string_or_none(value.get("message_id"))
                if not message_id:
                    continue
                ts = _parse_ts(value.get("seen_at"))
                if ts and ts < cutoff:
                    continue
                state["last_inbound_by_chat"][chat_key] = {
                    "message_id": message_id,
                    "space_id": _string_or_none(value.get("space_id")),
                    "seen_at": value.get("seen_at") if isinstance(value.get("seen_at"), str) else _now_iso(),
                }

        reactions = payload.get("reactions")
        if isinstance(reactions, dict):
            for key, value in reactions.items():
                if not isinstance(key, str) or not isinstance(value, dict):
                    continue
                created_ts = _parse_ts(value.get("created_at"))
                removed_ts = _parse_ts(value.get("removed_at"))
                newest_ts = max(created_ts, removed_ts)
                if newest_ts and newest_ts < cutoff:
                    continue
                state["reactions"][key] = {
                    "reaction_id": _string_or_none(value.get("reaction_id")),
                    "emoji": _string_or_none(value.get("emoji")) or "",
                    "created_at": value.get("created_at") if isinstance(value.get("created_at"), str) else _now_iso(),
                    "removed_at": value.get("removed_at") if isinstance(value.get("removed_at"), str) else None,
                }

        audit = payload.get("audit")
        if isinstance(audit, list):
            for item in audit:
                if not isinstance(item, dict):
                    continue
                at = item.get("at") if isinstance(item.get("at"), str) else _now_iso()
                state["audit"].append({
                    "at": at,
                    "action": _string_or_none(item.get("action")) or "unknown",
                    "status": _string_or_none(item.get("status")) or "unknown",
                    "chat_key": _string_or_none(item.get("chat_key")),
                    "message_id": _string_or_none(item.get("message_id")),
                    "reaction_id": _string_or_none(item.get("reaction_id")),
                    "error_class": _string_or_none(item.get("error_class")),
                    "error": _short_error(item.get("error")),
                })

        state["sent_messages"] = _trim_mapping(
            state["sent_messages"], self.sent_max, "sent_at"
        )
        state["last_inbound_by_chat"] = _trim_mapping(
            state["last_inbound_by_chat"], self.last_inbound_max, "seen_at"
        )
        state["reactions"] = _trim_mapping(
            state["reactions"], self.reactions_max, "created_at"
        )
        state["audit"] = state["audit"][-self.audit_max:]
        return state


def _string_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _trim_mapping(
    mapping: Dict[str, Dict[str, Any]], max_items: int, timestamp_key: str
) -> Dict[str, Dict[str, Any]]:
    if len(mapping) <= max_items:
        return dict(mapping)
    items = sorted(
        mapping.items(),
        key=lambda item: _parse_ts(item[1].get(timestamp_key)),
    )
    return dict(items[-max_items:])

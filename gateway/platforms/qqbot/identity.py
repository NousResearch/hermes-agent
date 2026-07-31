"""Stable sender identity and nickname observations for QQ group messages."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_MAX_NAME_LENGTH = 80


def _clean_name(value: Any) -> str:
    """Return a bounded, single-line value safe inside a sender prefix."""
    text = " ".join(str(value or "").split()).strip()
    text = text.replace("|", "／").replace("[", "【").replace("]", "】")
    return text[:_MAX_NAME_LENGTH]


def _stable_sender_id(group_openid: str, member_openid: str) -> str:
    raw = f"qqbot\0{group_openid}\0{member_openid}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:8]


@dataclass(frozen=True)
class QQSenderIdentity:
    """Resolved QQ sender identity for one inbound group message."""

    group_openid: str
    member_openid: str
    stable_id: str
    group_display_name: str = ""

    @property
    def label(self) -> str:
        parts = [f"QQ sender id={self.stable_id}"]
        if self.group_display_name:
            parts.append(f"群昵称={self.group_display_name}")
        return " | ".join(parts)


class QQIdentityStore:
    """Persist QQ nickname observations keyed by group + member OpenID."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {"version": 1, "members": {}}
        self._load()

    @staticmethod
    def member_key(group_openid: str, member_openid: str) -> str:
        return f"{group_openid}:{member_openid}"

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if not isinstance(loaded, dict) or not isinstance(loaded.get("members"), dict):
            return
        self._data = loaded
        self._data.setdefault("version", 1)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp_path = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temp_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, self.path)
        os.chmod(self.path, 0o600)

    def resolve(self, group_openid: str, author: dict[str, Any]) -> QQSenderIdentity:
        """Resolve the current sender and record its latest group display name."""
        member_openid = str(author.get("member_openid") or author.get("id") or "").strip()
        stable_id = _stable_sender_id(group_openid, member_openid)
        group_display_name = _clean_name(author.get("username"))
        key = self.member_key(group_openid, member_openid)

        with self._lock:
            members = self._data.setdefault("members", {})
            profile = members.setdefault(key, {})
            changed = profile.get("stable_id") != stable_id
            profile["stable_id"] = stable_id

            observed = profile.get("group_display_name")
            observed_value = observed.get("value") if isinstance(observed, dict) else ""
            if group_display_name and observed_value != group_display_name:
                profile["group_display_name"] = {
                    "value": group_display_name,
                    "source": "event.author.username",
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                }
                changed = True

            if changed:
                self._write()

        return QQSenderIdentity(
            group_openid=group_openid,
            member_openid=member_openid,
            stable_id=stable_id,
            group_display_name=group_display_name,
        )

"""Profile/bot-scoped ownership for Telegram widgets; never trust IDs from tool arguments."""

from __future__ import annotations

import hashlib
from pathlib import Path

from hermes_cli.plugins_state import PluginState


class WidgetStore(PluginState):
    """Use the plugin's atomic, quota-bounded state rather than a second database."""

    def __init__(self, home: Path, bot_id: str):
        super().__init__("telegram-platform")
        self._directory = Path(home) / "plugin-data" / "telegram-platform"
        self._bot_id = str(bot_id)

    @property
    def data_dir(self) -> Path:
        return self._directory

    def _key(self, kind: str, *parts: object) -> str:
        import json
        identity = json.dumps([self._bot_id, *map(str, parts)], separators=(",", ":"))
        return "widget:" + kind + ":" + hashlib.sha256(identity.encode()).hexdigest()

    def message(self, chat_id: str, message_id: str) -> dict | None:
        value = self.get(self._key("message", chat_id, message_id))
        if not isinstance(value, dict):
            return None
        if value.get("widget_id"):
            shared = self.shared(value["widget_id"])
            if shared:
                value = {**value, "public_data": shared.get("public_data")}
        return value

    def shared(self, widget_id: str) -> dict | None:
        value = self.get(self._key("shared", widget_id))
        return value if isinstance(value, dict) else None

    def save_message(self, record: dict) -> None:
        self.set(self._key("message", record["chat_id"], record["message_id"]), record)
        if record.get("widget_id"):
            self.set(self._key("shared", record["widget_id"]), record)

    @staticmethod
    def require_owner(record: dict | None, owner: dict) -> dict:
        if record is None:
            raise ValueError("Unknown widget; only widgets created by this Hermes session can be changed")
        for key in ("chat_id", "user_id", "session_key", "session_id"):
            if not owner.get(key) or str(record.get(key, "")) != str(owner[key]):
                raise ValueError("Widget belongs to another conversation or user")
        if str(record.get("thread_id") or "") != str(owner.get("thread_id") or ""):
            raise ValueError("Widget belongs to another topic")
        return record

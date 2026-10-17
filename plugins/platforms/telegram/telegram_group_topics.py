"""Online forum-topic learning and Telegram's channel-directory seam.

Telegram cannot enumerate old forum names. Service events rename; optional creation
replies only seed unknowns so stale reply metadata cannot undo a learned rename.
"""

from __future__ import annotations

import hashlib
import json
from plugins.platforms.telegram.telegram_group_topic_cache import TelegramGroupTopicCacheMixin, optional_cache_action


def configured_group_topics(value):
    """Yield (chat_id, chat_name, topic) from both supported config shapes."""
    if isinstance(value, dict):
        entries = ({"chat_id": key, "topics": topics} for key, topics in value.items())
    elif isinstance(value, list):
        entries = (entry for entry in value if isinstance(entry, dict))
    else:
        return
    for entry in entries:
        chat_id, topics = entry.get("chat_id"), entry.get("topics")
        if not isinstance(chat_id, (str, int)) or not str(chat_id).strip() or not isinstance(topics, list):
            continue
        chat_name = entry.get("chat_name") or entry.get("name") or str(chat_id)
        for topic in topics:
            if not isinstance(topic, dict) or not isinstance(topic.get("thread_id"), (str, int)):
                continue
            clean = {key: value for key, value in topic.items() if key in {"name", "skill"} and isinstance(value, str)}
            clean["thread_id"] = str(topic["thread_id"])
            if clean["thread_id"].strip():
                yield str(chat_id), str(chat_name), clean


def _session_origins(db, home, profile_name):
    """Current peer rows, with the same legacy JSON fallback as session discovery."""
    rows = db.list_gateway_sessions(platform="telegram", active_only=False)
    if not rows:
        try:
            data = json.loads((home / "sessions" / "sessions.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return
        rows = [{**entry, "origin_json": json.dumps(entry["origin"])}
                for entry in data.values() if isinstance(entry, dict) and isinstance(entry.get("origin"), dict)
                and entry["origin"].get("platform") == "telegram"] if isinstance(data, dict) else []
    for row in rows:
        try:
            origin = json.loads(row["origin_json"]) if row.get("origin_json") else {
                "chat_id": row.get("chat_id"), "thread_id": row.get("thread_id"), "chat_name": row.get("display_name"),
            }
        except (TypeError, ValueError):
            continue
        if isinstance(origin, dict) and (row.get("transport_profile") or row.get("profile_name") or "default") == profile_name:
            yield origin, row.get("chat_type") or origin.get("chat_type") or "dm"


class TelegramGroupTopicsMixin(TelegramGroupTopicCacheMixin):
    """Persistent cache owned by the receiving adapter, never the routed runtime profile."""

    def _group_topic_namespace(self) -> dict[str, str]:
        token = self.config.token or ""
        public_id, separator, _ = token.partition(":")
        # Prefer the stable public bot id (survives token rotation); malformed/test credentials
        # still get an isolated non-reversible identity, never a raw credential in SQLite/logs.
        bot_id = public_id if separator and public_id.isdecimal() else hashlib.sha256(token.encode()).hexdigest()
        profile = getattr(self, "_owner_profile", None) or getattr(self, "_hermes_profile_name", None) or "default"
        return {"profile_name": profile, "bot_id": bot_id}

    @staticmethod
    def _is_group_forum(message) -> bool:
        chat = message.chat
        return str(chat.type) in {"group", "supergroup"} and getattr(chat, "is_forum", False) is True

    async def _handle_forum_topic_status_update(self, update, context) -> None:
        """Textless service events learn labels and return, without entering the agent path."""
        message = self._effective_update_message(update)
        if message is None or not self._is_group_forum(message):
            return
        thread_id = getattr(message, "message_thread_id", None)
        if thread_id is None:
            return
        def learn(db):
            key = dict(**self._group_topic_namespace(), chat_id=str(message.chat.id), thread_id=str(thread_id))
            db.record_telegram_group_topic_observation(**key)
            for source in ("forum_topic_created", "forum_topic_edited"):
                name = getattr(getattr(message, source, None), "name", None)
                if isinstance(name, str) and name.strip():
                    db.set_telegram_group_topic_name(**key, name=name, source=source)
        self._optional_group_topic_cache(learn)
        self._accept_update()

    def _build_observed_message_event(self, message, msg_type, update_id=None):
        """Ingress-only builder: a display lookup is not proof this bot received traffic."""
        if self._is_group_forum(message):
            thread_id = self._effective_message_thread_id(message)
            if thread_id:
                self._optional_group_topic_cache(lambda db: self._observe_group_topic_message(db, message, thread_id))
        return self._build_message_event(message, msg_type, update_id=update_id)

    def _observe_group_topic_message(self, db, message, thread_id):
        key = dict(**self._group_topic_namespace(), chat_id=str(message.chat.id), thread_id=thread_id)
        db.record_telegram_group_topic_observation(**key)
        reply = getattr(message, "reply_to_message", None)
        created_name = getattr(getattr(reply, "forum_topic_created", None), "name", None)
        if isinstance(created_name, str) and created_name.strip():
            db.set_telegram_group_topic_name(**key, name=created_name, source="creation_reply", only_if_unknown=True)

    def _resolve_group_topic_binding(self, message, thread_id: str) -> tuple:
        name = skill = None
        for chat_id, _, topic in configured_group_topics(self.config.extra.get("group_topics")):
            if chat_id == str(message.chat.id) and str(topic["thread_id"]) == thread_id:
                name, skill = topic.get("name"), topic.get("skill")
                break
        if not self._is_group_forum(message):
            return name, skill
        def resolve(db):
            return name or db.get_telegram_group_topic_name(
                **self._group_topic_namespace(), chat_id=str(message.chat.id), thread_id=thread_id)
        return self._optional_group_topic_cache(resolve, fallback=name), skill

    async def list_channels(self) -> list[dict]:
        """Merge this transport's persisted sessions, learned labels and configured topics."""
        import asyncio
        return await asyncio.to_thread(self._list_group_topic_channels)

    def _list_group_topic_channels(self) -> list[dict]:
        fallback = self._merge_group_topic_channels([], [])
        def discover(db):
            return self._merge_group_topic_channels(
                _session_origins(db, self._update_receipt_dir, self._group_topic_namespace()["profile_name"]),
                optional_cache_action(lambda: db.list_telegram_group_topic_names(**self._group_topic_namespace()), []))
        return self._optional_group_topic_cache(discover, fallback=fallback)

    def _merge_group_topic_channels(self, sessions, topics) -> list[dict]:
        from gateway.channel_directory import _session_entry_id, _session_entry_name
        configured = list(configured_group_topics(self.config.extra.get("group_topics")))
        owned = {(topic["chat_id"], topic["thread_id"]) for topic in topics}
        explicit = {(chat_id, str(topic["thread_id"])) for chat_id, _, topic in configured}
        origins = {}
        for origin, kind in sessions:
            if not _session_entry_id(origin):
                continue
            # Persisted group labels may belong to a prior bot. Only this bot's scoped cache or
            # explicit config can name them; DM topic behavior stays untouched.
            if kind == "group" and origin.get("thread_id"):
                key = (str(origin.get("chat_id")), str(origin["thread_id"]))
                if key not in owned and key not in explicit:
                    continue
                origin["chat_topic"] = None
            origins.setdefault(_session_entry_id(origin), (origin, kind))
        chat_names = {str(origin["chat_id"]): origin.get("chat_name") for origin, _ in origins.values()}
        for topic in topics:
            entry_id = f"{topic['chat_id']}:{topic['thread_id']}"
            origin, kind = origins.get(entry_id, ({"chat_id": topic["chat_id"], "thread_id": topic["thread_id"]}, "group"))
            origin.update(chat_topic=topic["name"], chat_name=chat_names.get(topic["chat_id"]) or origin.get("chat_name"))
            origins[entry_id] = (origin, kind)
        for chat_id, chat_name, topic in configured:
            entry_id = f"{chat_id}:{topic['thread_id']}"
            origin, kind = origins.get(entry_id, ({"chat_id": chat_id, "thread_id": str(topic["thread_id"])}, "group"))
            origin["chat_name"] = chat_names.get(chat_id) if chat_name == chat_id and chat_names.get(chat_id) else chat_name
            if topic.get("name"):
                origin["chat_topic"] = topic["name"]
            origins[entry_id] = (origin, kind)
        return [{"id": entry_id, "name": _session_entry_name(origin), "type": kind,
                 "thread_id": origin.get("thread_id")} for entry_id, (origin, kind) in origins.items()]

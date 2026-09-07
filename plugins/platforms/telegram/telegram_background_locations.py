"""RAM-only handling for Telegram live-location context.

Coordinates are owned by the live adapter, never by a profile file or a queued
event. A foreground event carries only an opaque reference; the runner resolves
it once, immediately before its agent turn starts.
"""
from __future__ import annotations
import hashlib, logging, math, secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from gateway.platforms.event import MessageEvent

logger = logging.getLogger(__name__)
_MAX_BACKGROUND_LOCATION_SUBJECTS = 512

@dataclass(frozen=True)
class TelegramLiveLocationRef:
    """Non-serializable capability for one adapter-local location lookup."""
    adapter_id: str
    bot_scope: str
    subject_key: str
    profile: str
    profile_incarnation: Optional[tuple[int, int, int]]

class TelegramBackgroundLocationsMixin:
    _BACKGROUND_LOCATION_CONTEXT_HEADER = "[Background Telegram location context]"
    _BACKGROUND_LOCATION_INDEFINITE_LIVE_PERIOD = 0x7FFFFFFF
    _BACKGROUND_LOCATION_MAX_SUBJECTS = _MAX_BACKGROUND_LOCATION_SUBJECTS

    def _init_background_locations(self, config: Any) -> None:
        raw = self.config.extra.get("background_locations") if getattr(self.config, "extra", None) else None
        if isinstance(raw, bool): enabled = raw
        elif isinstance(raw, str):
            normalized = raw.strip().lower(); enabled = normalized in {"true", "1", "yes", "on"}
            if normalized not in {"true", "1", "yes", "on", "false", "0", "no", "off", ""}: logger.warning("[Telegram] Ignoring invalid background_locations value; explicit true/false is required")
        else:
            enabled = False
            if raw is not None: logger.warning("[Telegram] Ignoring non-scalar background_locations value; explicit true/false is required")
        self._background_locations_configured = enabled
        self._background_locations_enabled = enabled
        self._background_location_bot_scope = self._resolve_background_location_bot_scope(getattr(config, "token", None))
        self._background_location_adapter_id = secrets.token_urlsafe(18)
        self._background_location_records: Dict[str, dict] = {}
        # Coordinate-free terminal lifecycle markers reject delayed edits after
        # Telegram says a share stopped or the finite share expires.
        self._background_location_terminal_updates: Dict[tuple[str, str, str], datetime] = {}

    @staticmethod
    def _resolve_background_location_bot_scope(token: Any) -> str:
        raw = str(token or ""); bot_id, separator, _ = raw.partition(":")
        return bot_id if separator and bot_id.isdigit() else "token-" + hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _coerce_finite_float(value: Any, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> Optional[float]:
        if isinstance(value, bool): return None
        try: number = float(value)
        except (TypeError, ValueError, OverflowError): return None
        if not math.isfinite(number) or (minimum is not None and number < minimum) or (maximum is not None and number > maximum): return None
        return number

    @staticmethod
    def _coerce_nonnegative_int(value: Any) -> Optional[int]:
        if isinstance(value, bool): return None
        try: number = int(value)
        except (TypeError, ValueError, OverflowError): return None
        return number if number >= 0 else None

    def _active_live_location_period(self, location: Any) -> Optional[int]:
        period = self._coerce_nonnegative_int(getattr(location, "live_period", None))
        return period if period and period <= self._BACKGROUND_LOCATION_INDEFINITE_LIVE_PERIOD else None

    @staticmethod
    def _is_background_location_edited_update(update: Any) -> bool:
        return bool(getattr(update, "edited_message", None) or getattr(update, "edited_channel_post", None) or getattr(update, "edited_business_message", None))

    @staticmethod
    def _is_background_location_business_update(update: Any, message: Any) -> bool:
        return bool(getattr(message, "business_connection_id", None) is not None or getattr(update, "business_message", None) or getattr(update, "edited_business_message", None))

    def _is_background_live_location_update(self, update: Any, message: Any) -> bool:
        if getattr(message, "venue", None) is not None or getattr(message, "location", None) is None: return False
        return self._active_live_location_period(message.location) is not None or self._is_background_location_edited_update(update)

    def _background_location_source_for_message(self, message: Any) -> Optional[Any]:
        if getattr(message, "business_connection_id", None) is not None or getattr(getattr(message, "sender_chat", None), "id", None) is not None: return None
        source = self._source_from_message_for_auth(message)
        resolver = getattr(getattr(self, "gateway_runner", None), "_profile_name_for_source", None)
        if callable(resolver) and not getattr(source, "profile", None):
            try: source.profile = resolver(source)
            except Exception: logger.warning("[Telegram] Could not resolve background location profile", exc_info=True); return None
        if not getattr(source, "profile", None): source.profile = str(getattr(self, "_owner_profile", "") or "").strip() or None
        return source

    def _background_location_subject_key_from_source(self, source: Any) -> Optional[str]:
        user_id, chat_id = getattr(source, "user_id", None), getattr(source, "chat_id", None)
        if user_id is None or chat_id is None: return None
        key = f"bot:{self._background_location_bot_scope}:chat:{chat_id}:user:{user_id}"
        if str(getattr(source, "chat_type", "") or "").lower() not in {"private", "dm"} and getattr(source, "thread_id", None) is not None: key += f":thread:{source.thread_id}"
        return key

    def _background_location_subject_key(self, message: Any) -> Optional[str]: return self._background_location_subject_key_from_source(self._background_location_source_for_message(message))

    def _profile_incarnation(self, profile: str) -> Optional[tuple[int, int, int]]:
        try:
            if profile:
                from hermes_cli.profiles import get_profile_dir
                path = Path(get_profile_dir(profile))
            else:
                from hermes_constants import get_hermes_home
                path = Path(get_hermes_home())
            stat = path.stat(); return stat.st_dev, stat.st_ino, stat.st_ctime_ns
        except (OSError, ValueError): return None

    def _background_location_lifecycle_identity(self, message: Any) -> Optional[tuple[str, str, str]]:
        chat = str(getattr(getattr(message, "chat", None), "id", "") or ""); user = str(getattr(getattr(message, "from_user", None), "id", "") or ""); message_id = str(getattr(message, "message_id", "") or "")
        return (chat, user, message_id) if chat and user and message_id else None

    def _is_background_location_authorized(self, message: Any) -> bool:
        source = self._source_from_message_for_auth(message)
        return self._is_sender_authorized(source.user_id, source.chat_type, source.chat_id, is_bot=source.is_bot, thread_id=source.thread_id) is True

    def _should_accept_background_location(self, message: Any) -> bool:
        if self._is_own_message(message): return False
        thread_id = self._effective_message_thread_id(message)
        if thread_id is not None:
            try:
                if int(thread_id) in self._telegram_ignored_threads(): return False
            except (TypeError, ValueError): return False
        if not self._is_group_chat(message): return True
        topics = self._telegram_allowed_topics()
        if topics and str(thread_id if thread_id is not None else self._GENERAL_TOPIC_THREAD_ID) not in topics: return False
        chats = self._telegram_allowed_chats()
        return not chats or str(getattr(getattr(message, "chat", None), "id", "")) in chats

    @staticmethod
    def _timestamp(value: Any) -> Optional[datetime]:
        if not isinstance(value, datetime): return None
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)

    def _is_newer(self, previous: dict, timestamp: Optional[datetime], update_id: Optional[int]) -> bool:
        previous_time, previous_id = previous.get("telegram_timestamp"), previous.get("update_id")
        if timestamp is not None and previous_time is not None:
            if timestamp != previous_time: return timestamp > previous_time
            return update_id is not None and (previous_id is None or update_id > previous_id)
        return update_id is None or previous_id is None or update_id > previous_id

    def _prune_expired_background_locations(self) -> None:
        now = datetime.now(timezone.utc)
        active = {}
        for key, record in self._background_location_records.items():
            if record.get("expires_at") is None or now < record["expires_at"]:
                active[key] = record
            else:
                self._background_location_terminal_updates[record["lifecycle"]] = now
        self._background_location_records = active
        if len(self._background_location_terminal_updates) > self._BACKGROUND_LOCATION_MAX_SUBJECTS:
            oldest = min(self._background_location_terminal_updates, key=self._background_location_terminal_updates.get)
            self._background_location_terminal_updates.pop(oldest, None)

    async def _prepare_background_locations_for_connect(self) -> Dict[str, dict]:
        # A reconnect cannot prove it received every stop edit while down.
        # RAM-only retention therefore fails closed at each connection boundary.
        self._background_location_records.clear(); self._background_location_terminal_updates.clear(); return {}

    async def _persist_background_location(self, update: Any, message: Any, *, polling_admission: Any = None) -> bool:
        del polling_admission; self._prune_expired_background_locations()
        identity, subject = self._background_location_lifecycle_identity(message), self._background_location_subject_key(message)
        if identity is None or subject is None: return False
        period, edited = self._active_live_location_period(getattr(message, "location", None)), self._is_background_location_edited_update(update)
        if edited and period is None:
            self._background_location_terminal_updates[identity] = datetime.now(timezone.utc)
            if len(self._background_location_terminal_updates) > self._BACKGROUND_LOCATION_MAX_SUBJECTS:
                oldest = min(self._background_location_terminal_updates, key=self._background_location_terminal_updates.get)
                self._background_location_terminal_updates.pop(oldest, None)
            for key, record in tuple(self._background_location_records.items()):
                if record.get("lifecycle") == identity: self._background_location_records.pop(key, None)
            return True
        if period is None: return False
        if identity in self._background_location_terminal_updates: return True
        location = message.location
        latitude = self._coerce_finite_float(getattr(location, "latitude", None), minimum=-90, maximum=90); longitude = self._coerce_finite_float(getattr(location, "longitude", None), minimum=-180, maximum=180)
        if latitude is None or longitude is None: return False
        timestamp, update_id = self._timestamp(getattr(message, "edit_date", None) or getattr(message, "date", None)), self._coerce_nonnegative_int(getattr(update, "update_id", None))
        previous = self._background_location_records.get(subject)
        if previous and previous.get("lifecycle") == identity and not self._is_newer(previous, timestamp, update_id): return True
        started = self._timestamp(getattr(message, "date", None)) or datetime.now(timezone.utc); expires = None if period == self._BACKGROUND_LOCATION_INDEFINITE_LIVE_PERIOD else started + timedelta(seconds=period)
        source = self._background_location_source_for_message(message)
        self._background_location_records[subject] = {"lifecycle": identity, "latitude": latitude, "longitude": longitude, "recorded_at": datetime.now(timezone.utc), "telegram_timestamp": timestamp, "update_id": update_id, "expires_at": expires, "profile": str(getattr(source, "profile", "") or "")}
        if len(self._background_location_records) > self._BACKGROUND_LOCATION_MAX_SUBJECTS: self._background_location_records.pop(min(self._background_location_records, key=lambda key: self._background_location_records[key]["recorded_at"]), None)
        return True

    def _location_context(self, record: dict) -> str:
        recorded_at = record.get("telegram_timestamp") or record["recorded_at"]
        return "\n".join([self._BACKGROUND_LOCATION_CONTEXT_HEADER, "Source: live_location", "This is the latest snapshot of an active live location share, not a fixed one-time pin.", "The recorded position may be stale; use it only when relevant to the user's explicit request.", f"Recorded at (UTC): {recorded_at.isoformat()}", f"Latitude: {record['latitude']}", f"Longitude: {record['longitude']}"])

    async def _attach_background_location_context(self, event: MessageEvent, message: Any) -> MessageEvent:
        if not self._background_locations_enabled or event.internal: return event
        source = self._background_location_source_for_message(message); subject = self._background_location_subject_key_from_source(source)
        if source is None or subject is None: return event
        profile = str(getattr(source, "profile", "") or "")
        event.ephemeral_context_ref = TelegramLiveLocationRef(self._background_location_adapter_id, self._background_location_bot_scope, subject, profile, self._profile_incarnation(profile))
        return event

    def _resolve_ephemeral_user_context_for_dispatch_sync(self, event: MessageEvent) -> Optional[str]:
        ref = getattr(event, "ephemeral_context_ref", None)
        if getattr(event, "_ephemeral_context_refresh_unsafe", False): return None
        if not isinstance(ref, TelegramLiveLocationRef) or not self._background_locations_enabled: return None
        if ref.adapter_id != self._background_location_adapter_id or ref.bot_scope != self._background_location_bot_scope or ref.profile_incarnation != self._profile_incarnation(ref.profile): return None
        self._prune_expired_background_locations(); record = self._background_location_records.get(ref.subject_key)
        return self._location_context(record) if record and record.get("profile") == ref.profile else None

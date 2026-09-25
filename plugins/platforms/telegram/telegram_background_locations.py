"""RAM-only handling for Telegram live-location context.

The adapter owns the latest active coordinates. Foreground events carry only an
opaque, adapter-local reference; the gateway resolves that reference once when the
turn is admitted and binds an immutable snapshot to that turn.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import secrets
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from gateway.authz_mixin import _coerce_allow_set
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import is_shared_multi_user_session

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TelegramLiveLocationRef:
    """Non-serializable capability for one adapter-local location lookup."""

    adapter_id: str
    bot_scope: str
    subject_key: str
    profile: str
    generation: int


class TelegramBackgroundLocationsMixin:
    """Keep active Telegram live locations in bounded process memory."""

    _BACKGROUND_LOCATION_CONTEXT_HEADER = "[Background Telegram location context]"
    _BACKGROUND_LOCATION_INDEFINITE_LIVE_PERIOD = 0x7FFFFFFF
    _BACKGROUND_LOCATION_MAX_SUBJECTS = 512
    _BACKGROUND_LOCATION_MAX_UPDATE_RECEIPTS = 4096

    def _init_background_locations(self, config: Any) -> None:
        extra = getattr(getattr(self, "config", None), "extra", None) or {}
        raw = extra.get("background_locations")
        enabled = False
        if isinstance(raw, bool):
            enabled = raw
        elif isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                enabled = True
            elif normalized not in {"false", "0", "no", "off", ""}:
                logger.warning(
                    "[Telegram] Ignoring invalid background_locations value; "
                    "explicit true/false is required"
                )
        elif raw is not None:
            logger.warning(
                "[Telegram] Ignoring non-scalar background_locations value; "
                "explicit true/false is required"
            )

        self._background_locations_configured = enabled
        self._background_locations_enabled = enabled
        self._background_location_bot_scope = self._background_location_bot_identity(
            getattr(config, "token", None)
        )
        self._background_location_adapter_id = secrets.token_urlsafe(18)
        self._background_location_generation = 0
        self._background_location_polling_generation: Optional[int] = None
        self._background_location_polling_ready_generation: Optional[int] = None
        self._background_location_update_receipts: "OrderedDict[int, tuple[int, bool]]" = (
            OrderedDict()
        )
        self._background_location_expiry_handle: Optional[asyncio.TimerHandle] = None
        self._background_location_records: "OrderedDict[str, dict]" = OrderedDict()
        # Coordinate-free terminal markers prevent a delayed edit from resurrecting a
        # share after Telegram reported it stopped or expired.
        self._background_location_terminal_updates: "OrderedDict[tuple[str, str, str], datetime]" = (
            OrderedDict()
        )

    @staticmethod
    def _background_location_bot_identity(token: Any) -> str:
        raw = str(token or "")
        bot_id, separator, _ = raw.partition(":")
        if separator and bot_id.isdigit():
            return bot_id
        return "token-" + hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _coerce_finite_float(
        value: Any, *, minimum: Optional[float] = None, maximum: Optional[float] = None,
    ) -> Optional[float]:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number):
            return None
        if minimum is not None and number < minimum:
            return None
        if maximum is not None and number > maximum:
            return None
        return number

    @staticmethod
    def _coerce_nonnegative_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, timedelta):
            seconds = value.total_seconds()
            if not math.isfinite(seconds) or seconds < 0 or not seconds.is_integer():
                return None
            return int(seconds)
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if number >= 0 else None

    def _active_live_location_period(self, location: Any) -> Optional[int]:
        period = self._coerce_nonnegative_int(getattr(location, "live_period", None))
        if period and period <= self._BACKGROUND_LOCATION_INDEFINITE_LIVE_PERIOD:
            return period
        return None

    @staticmethod
    def _is_background_location_edited_update(update: Any) -> bool:
        return bool(
            getattr(update, "edited_message", None)
            or getattr(update, "edited_channel_post", None)
            or getattr(update, "edited_business_message", None)
        )

    @staticmethod
    def _is_background_location_business_update(update: Any, message: Any) -> bool:
        return bool(
            getattr(message, "business_connection_id", None) is not None
            or getattr(update, "business_message", None)
            or getattr(update, "edited_business_message", None)
        )

    def _is_background_live_location_update(self, update: Any, message: Any) -> bool:
        if getattr(message, "venue", None) is not None or getattr(message, "location", None) is None:
            return False
        # Any explicit live_period is lifecycle traffic, even when malformed. Edited
        # location messages without a period are Telegram's stop notification; treating
        # an unfamiliar edit as ordinary content would persist coordinates fail-open.
        return (
            getattr(message.location, "live_period", None) is not None
            or self._is_background_location_edited_update(update)
        )

    def _background_location_source_for_message(self, message: Any) -> Optional[Any]:
        if (
            getattr(message, "business_connection_id", None) is not None
            or getattr(getattr(message, "sender_chat", None), "id", None) is not None
        ):
            return None
        source = self._source_from_message_for_auth(message)
        # Use the exact canonical topic identity as foreground event construction.
        # In particular, Telegram forum General arrives with a raw None but routes as
        # thread "1"; leaving it unnormalized splits the live record from its turn.
        source.thread_id = self._effective_message_thread_id(message)
        resolver = getattr(getattr(self, "gateway_runner", None), "_profile_name_for_source", None)
        if callable(resolver) and not getattr(source, "profile", None):
            try:
                source.profile = resolver(source)
            except Exception:
                logger.warning(
                    "[Telegram] Could not resolve background location profile",
                    exc_info=True,
                )
                return None
        if not getattr(source, "profile", None):
            source.profile = str(getattr(self, "_owner_profile", "") or "").strip() or None
        return source

    def _background_location_subject_key_from_source(self, source: Any) -> Optional[str]:
        if source is None:
            return None
        user_id = getattr(source, "user_id", None)
        chat_id = getattr(source, "chat_id", None)
        if user_id is None or chat_id is None:
            return None
        key = f"bot:{self._background_location_bot_scope}:chat:{chat_id}:user:{user_id}"
        if getattr(source, "thread_id", None) is not None:
            key += f":thread:{source.thread_id}"
        return key

    def _background_location_subject_key(self, message: Any) -> Optional[str]:
        return self._background_location_subject_key_from_source(
            self._background_location_source_for_message(message)
        )

    @staticmethod
    def _timestamp(value: Any) -> Optional[datetime]:
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _background_location_lifecycle_identity(
        message: Any,
    ) -> Optional[tuple[str, str, str]]:
        chat = str(getattr(getattr(message, "chat", None), "id", "") or "")
        user = str(getattr(getattr(message, "from_user", None), "id", "") or "")
        message_id = str(getattr(message, "message_id", "") or "")
        return (chat, user, message_id) if chat and user and message_id else None

    def _is_background_location_authorized(self, message: Any) -> bool:
        source = self._source_from_message_for_auth(message)
        source.thread_id = self._effective_message_thread_id(message)
        user_id = getattr(source, "user_id", None)
        if not user_id:
            return False

        # Match Telegram's ordinary intake precedence: an adapter-scoped allow_from
        # is the sole authority when configured. The generic runner may otherwise
        # grant a paired user or every member of an allowed group, but neither may
        # bypass this explicit, narrower location-collection boundary.
        extra = getattr(getattr(self, "config", None), "extra", None) or {}
        allow_key = (
            "group_allow_from"
            if (getattr(source, "chat_type", "") or "")
            in {"group", "forum", "channel"}
            else "allow_from"
        )
        adapter_allow_from = extra.get(allow_key)
        if adapter_allow_from is not None:
            allowed = _coerce_allow_set(adapter_allow_from)
            return str(user_id) in allowed or "*" in allowed

        if getattr(self, "_authorization_check", None) is not None:
            return self._is_sender_authorized(
                user_id,
                source.chat_type,
                source.chat_id,
                is_bot=source.is_bot,
                thread_id=source.thread_id,
            ) is True
        auth_fn = self._legacy_runner_auth_fn()
        if auth_fn is not None:
            try:
                return bool(auth_fn(source))
            except Exception:
                return False
        decision = self._env_allowlist_decision(str(user_id))
        return decision is True

    def _should_accept_background_location(self, message: Any) -> bool:
        if self._is_own_message(message):
            return False
        thread_id = self._effective_message_thread_id(message)
        if thread_id is not None:
            try:
                if int(thread_id) in self._telegram_ignored_threads():
                    return False
            except (TypeError, ValueError):
                return False
        if not self._is_group_chat(message):
            return True
        topics = self._telegram_allowed_topics()
        if topics and str(
            thread_id if thread_id is not None else self._GENERAL_TOPIC_THREAD_ID
        ) not in topics:
            return False
        chats = self._telegram_allowed_chats()
        return not chats or str(getattr(getattr(message, "chat", None), "id", "")) in chats

    @staticmethod
    def _update_is_newer(
        previous: dict, timestamp: Optional[datetime], update_id: Optional[int],
    ) -> bool:
        previous_time = previous.get("telegram_timestamp")
        previous_id = previous.get("update_id")
        if timestamp is not None and previous_time is not None:
            if timestamp != previous_time:
                return timestamp > previous_time
            if update_id is None:
                return False
            return previous_id is None or update_id > previous_id
        if update_id is None:
            return previous_id is None and timestamp is not None and previous_time is None
        return previous_id is None or update_id > previous_id

    def _remember_terminal_location(
        self, identity: tuple[str, str, str], observed_at: datetime,
    ) -> None:
        terminals = self._background_location_terminal_updates
        terminals[identity] = observed_at
        terminals.move_to_end(identity)
        while len(terminals) > self._BACKGROUND_LOCATION_MAX_SUBJECTS:
            terminals.popitem(last=False)

    def _prune_expired_background_locations(self) -> None:
        now = datetime.now(timezone.utc)
        for key, record in tuple(self._background_location_records.items()):
            expires_at = record.get("expires_at")
            if expires_at is not None and now >= expires_at:
                self._background_location_records.pop(key, None)
                self._remember_terminal_location(record["lifecycle"], now)

    def _cancel_background_location_expiry_timer(self) -> None:
        handle = getattr(self, "_background_location_expiry_handle", None)
        self._background_location_expiry_handle = None
        if handle is not None:
            handle.cancel()

    def _schedule_background_location_expiry(self) -> None:
        """Arm one adapter-owned timer for the nearest finite share expiry."""
        self._cancel_background_location_expiry_timer()
        finite_expiries = [
            record.get("expires_at")
            for record in self._background_location_records.values()
            if record.get("expires_at") is not None
        ]
        if not finite_expiries:
            return
        deadline = min(finite_expiries)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Recording happens on PTB's event loop in production. A synchronous test
            # double can still rely on opportunistic pruning without leaking to disk.
            return

        delay = max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())

        def _expire() -> None:
            if self._background_location_expiry_handle is not handle:
                return
            self._background_location_expiry_handle = None
            self._prune_expired_background_locations()
            self._schedule_background_location_expiry()

        handle = loop.call_later(delay, _expire)
        self._background_location_expiry_handle = handle

    def _observe_background_location_poll_result(
        self, generation: int, envelope: Any,
    ) -> None:
        """Stamp update ids with receive-generation proof and detect a drained queue.

        Updates returned before the first empty response are backlog and remain
        permanently ineligible even if their handlers run after the empty response.
        """
        if generation != getattr(self, "_background_location_polling_generation", None):
            return
        results = envelope.get("result") if isinstance(envelope, dict) else None
        if not isinstance(results, list):
            return
        if not results:
            self._background_location_polling_ready_generation = generation
            return
        ready = (
            getattr(self, "_background_location_polling_ready_generation", None)
            == generation
        )
        receipts = self._background_location_update_receipts
        for raw_update in results:
            if not isinstance(raw_update, dict):
                continue
            update_id = self._coerce_nonnegative_int(raw_update.get("update_id"))
            if update_id is None:
                continue
            receipts[update_id] = (generation, ready)
            receipts.move_to_end(update_id)
        while len(receipts) > self._BACKGROUND_LOCATION_MAX_UPDATE_RECEIPTS:
            receipts.popitem(last=False)

    def _background_location_active_update_is_admissible(self, update: Any) -> bool:
        """Whether an active update was fetched after this generation drained."""
        generation = getattr(self, "_background_location_polling_generation", None)
        if generation is None:
            # Before connect there is no network delivery path; this keeps the pure
            # record/handler unit boundary usable without weakening production polling.
            return True
        update_id = self._coerce_nonnegative_int(getattr(update, "update_id", None))
        return bool(
            update_id is not None
            and self._background_location_update_receipts.get(update_id)
            == (generation, True)
        )

    def _clear_background_locations(self) -> None:
        # Rotate the capability generation before clearing. A queued event from the old
        # receive epoch must never resolve a new record that happens to reuse its subject.
        self._background_location_generation = (
            getattr(self, "_background_location_generation", 0) + 1
        )
        self._background_location_polling_ready_generation = None
        receipts = getattr(self, "_background_location_update_receipts", None)
        if receipts is not None:
            receipts.clear()
        self._cancel_background_location_expiry_timer()
        records = getattr(self, "_background_location_records", None)
        terminals = getattr(self, "_background_location_terminal_updates", None)
        if records is not None:
            records.clear()
        if terminals is not None:
            terminals.clear()

    def _begin_background_location_polling_generation(self, generation: int) -> None:
        self._clear_background_locations()
        if not getattr(self, "_background_locations_configured", False):
            self._background_location_polling_generation = None
            return
        self._background_location_polling_generation = generation

    async def _prepare_background_locations_for_connect(self) -> None:
        # A reconnect cannot prove it observed every stop while disconnected.
        self._clear_background_locations()

    async def _record_background_location(self, update: Any, message: Any) -> bool:
        """Apply one active/edit/stop update to bounded adapter-local RAM."""
        self._prune_expired_background_locations()
        identity = self._background_location_lifecycle_identity(message)
        if identity is None:
            return False

        location = getattr(message, "location", None)
        period = self._active_live_location_period(location)
        edited = self._is_background_location_edited_update(update)
        if edited and period is None:
            # A stop's Telegram lifecycle identity is sufficient to revoke the old
            # share. Purge before profile routing: a transient resolver failure must
            # never leave an indefinite location active after Telegram stopped it.
            self._remember_terminal_location(identity, datetime.now(timezone.utc))
            for key, record in tuple(self._background_location_records.items()):
                if record.get("lifecycle") == identity:
                    self._background_location_records.pop(key, None)
            self._schedule_background_location_expiry()
            return True
        if period is None:
            return False
        source = self._background_location_source_for_message(message)
        subject = self._background_location_subject_key_from_source(source)
        if source is None or subject is None:
            return False
        if identity in self._background_location_terminal_updates:
            return True
        latitude = self._coerce_finite_float(
            getattr(location, "latitude", None), minimum=-90, maximum=90
        )
        longitude = self._coerce_finite_float(
            getattr(location, "longitude", None), minimum=-180, maximum=180
        )
        if latitude is None or longitude is None:
            return False
        timestamp = self._timestamp(
            getattr(message, "edit_date", None) or getattr(message, "date", None)
        )
        update_id = self._coerce_nonnegative_int(getattr(update, "update_id", None))
        previous = self._background_location_records.get(subject)
        if previous and not self._update_is_newer(previous, timestamp, update_id):
            return True

        started_at = self._timestamp(getattr(message, "date", None)) or datetime.now(timezone.utc)
        expires_at = (
            None
            if period == self._BACKGROUND_LOCATION_INDEFINITE_LIVE_PERIOD
            else started_at + timedelta(seconds=period)
        )
        profile = str(getattr(source, "profile", "") or "")
        self._background_location_records[subject] = {
            "lifecycle": identity,
            "latitude": latitude,
            "longitude": longitude,
            "recorded_at": datetime.now(timezone.utc),
            "telegram_timestamp": timestamp,
            "update_id": update_id,
            "expires_at": expires_at,
            "profile": profile,
        }
        self._background_location_records.move_to_end(subject)
        while len(self._background_location_records) > self._BACKGROUND_LOCATION_MAX_SUBJECTS:
            self._background_location_records.popitem(last=False)
        self._schedule_background_location_expiry()
        return True

    def _location_context(self, record: dict) -> str:
        recorded_at = record.get("telegram_timestamp") or record["recorded_at"]
        return "\n".join(
            [
                self._BACKGROUND_LOCATION_CONTEXT_HEADER,
                "Source: live_location",
                "This is one immutable snapshot of an active live location share, not a fixed pin.",
                "It may be stale; use it only when relevant to the user's explicit request.",
                f"Recorded at (UTC): {recorded_at.isoformat()}",
                f"Latitude: {record['latitude']}",
                f"Longitude: {record['longitude']}",
            ]
        )

    def _background_location_session_is_shared(self, source: Any) -> bool:
        extra = getattr(getattr(self, "config", None), "extra", None) or {}
        return is_shared_multi_user_session(
            source,
            group_sessions_per_user=extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=extra.get("thread_sessions_per_user", False),
        )

    async def _attach_background_location_context(
        self, event: MessageEvent, message: Any,
    ) -> MessageEvent:
        """Attach a coordinate-free capability to an eligible foreground text turn."""
        if (
            not getattr(self, "_background_locations_enabled", False)
            or getattr(self, "_send_path_degraded", False)
            or event.internal
            or event.message_type not in {MessageType.TEXT, MessageType.COMMAND}
            or not (event.text or "").strip()
            or bool(event.media_urls)
            or getattr(event, "_ephemeral_context_blocked", False)
        ):
            return event
        source = self._background_location_source_for_message(message)
        subject = self._background_location_subject_key_from_source(source)
        event_source = getattr(event, "source", None)
        if (
            source is None
            or subject is None
            or event_source is None
            or self._background_location_session_is_shared(event_source)
            or self._background_location_subject_key_from_source(event_source) != subject
            or str(getattr(event_source, "profile", "") or "")
            != str(getattr(source, "profile", "") or "")
        ):
            return event
        self._prune_expired_background_locations()
        if subject not in self._background_location_records:
            return event
        profile = str(getattr(source, "profile", "") or "")
        event.ephemeral_context_ref = TelegramLiveLocationRef(
            adapter_id=self._background_location_adapter_id,
            bot_scope=self._background_location_bot_scope,
            subject_key=subject,
            profile=profile,
            generation=self._background_location_generation,
        )
        return event

    def _resolve_ephemeral_user_context_for_dispatch_sync(
        self, event: MessageEvent,
    ) -> Optional[str]:
        """Resolve an opaque capability once at the foreground-turn boundary."""
        ref = getattr(event, "ephemeral_context_ref", None)
        if (
            not isinstance(ref, TelegramLiveLocationRef)
            or not getattr(self, "_background_locations_enabled", False)
            or getattr(self, "_send_path_degraded", False)
            or getattr(self, "_teardown_started", False)
            or getattr(event, "_ephemeral_context_blocked", False)
        ):
            return None
        if (
            ref.adapter_id != self._background_location_adapter_id
            or ref.bot_scope != self._background_location_bot_scope
            or ref.generation != self._background_location_generation
        ):
            return None
        source = getattr(event, "source", None)
        if source is None or self._background_location_session_is_shared(source):
            return None
        if self._background_location_subject_key_from_source(source) != ref.subject_key:
            return None
        if str(getattr(source, "profile", "") or "") != ref.profile:
            return None
        self._prune_expired_background_locations()
        record = self._background_location_records.get(ref.subject_key)
        if not record or record.get("profile") != ref.profile:
            return None
        return self._location_context(record)

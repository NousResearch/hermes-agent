"""Nostr gateway adapter for NIP-17 direct messages."""

import asyncio
import importlib
import importlib.util
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource

logger = logging.getLogger(__name__)

NIP17_DM_KIND = 14
GIFT_WRAP_KIND = 1059
DM_RELAY_LIST_KIND = 10050
POLL_INTERVAL_SECONDS = 2.0
FETCH_TIMEOUT = timedelta(seconds=5)
FETCH_LIMIT = 50
MAX_SEEN_EVENT_IDS = 512


def check_nostr_requirements(config: Optional[PlatformConfig] = None) -> bool:
    """Return True when Nostr is configured and the Python SDK is installed."""
    has_sdk = importlib.util.find_spec("nostr_sdk") is not None
    if not has_sdk:
        return False

    if config is not None:
        secret_key = config.token or (config.extra or {}).get("secret_key", "")
        relays = _parse_relay_list((config.extra or {}).get("relays", []))
    else:
        secret_key = os.getenv("NOSTR_SECRET_KEY") or os.getenv("NOSTR_NSEC")
        relays = _parse_relay_list(os.getenv("NOSTR_RELAYS", ""))

    return bool(secret_key and relays)


def _load_nostr_sdk():
    """Import nostr_sdk lazily so tests can run without the extra installed."""
    return importlib.import_module("nostr_sdk")


def _parse_relay_list(raw_relays: str | Iterable[str]) -> List[str]:
    """Normalize relay URLs from env/config into a deduplicated list."""
    if isinstance(raw_relays, str):
        items = raw_relays.split(",")
    else:
        items = list(raw_relays or [])

    relays: List[str] = []
    seen: set[str] = set()
    for item in items:
        relay = str(item).strip()
        if not relay or relay in seen:
            continue
        seen.add(relay)
        relays.append(relay)
    return relays


def normalize_nostr_identifier(value: str) -> str:
    """Normalize a Nostr public identifier to lowercase hex when possible."""
    normalized = str(value or "").strip()
    if not normalized:
        return ""
    try:
        sdk = _load_nostr_sdk()
        return _public_key_to_hex(sdk.PublicKey.parse(normalized)).lower()
    except Exception:
        return normalized.lower()


def _event_to_dict(event: Any) -> Dict[str, Any]:
    """Convert an SDK event-like object into a plain dict."""
    if isinstance(event, dict):
        return event
    if hasattr(event, "as_json"):
        return json.loads(event.as_json())
    raise TypeError(f"Unsupported event type: {type(event)!r}")


def _iter_events(events: Any) -> List[Any]:
    """Normalize an SDK event collection into a list."""
    if events is None:
        return []
    if isinstance(events, list):
        return events
    if hasattr(events, "to_vec"):
        return list(events.to_vec())
    try:
        return list(events)
    except TypeError:
        return []


def _iter_valid_event_pairs(events: Any) -> List[tuple[Any, Dict[str, Any]]]:
    """Return `(event, event_data)` pairs, skipping malformed event payloads."""
    pairs: List[tuple[Any, Dict[str, Any]]] = []
    for event in _iter_events(events):
        try:
            pairs.append((event, _event_to_dict(event)))
        except Exception as exc:
            logger.debug("Nostr: ignoring malformed event payload: %s", exc)
    return pairs


def _event_created_at(event_data: Dict[str, Any]) -> int:
    value = event_data.get("created_at")
    if isinstance(value, int):
        return value
    return 0


def _public_key_to_hex(public_key: Any) -> str:
    """Best-effort conversion of a PublicKey-like object into hex."""
    if hasattr(public_key, "to_hex"):
        return str(public_key.to_hex())
    return str(public_key)


def _public_key_to_bech32(public_key: Any) -> str:
    """Best-effort conversion of a PublicKey-like object into npub."""
    if hasattr(public_key, "to_bech32"):
        return str(public_key.to_bech32())
    return str(public_key)


def _message_event_from_rumor(
    *,
    sender_hex: str,
    sender_display: str,
    rumor_data: Dict[str, Any],
    raw_message: Dict[str, Any],
) -> MessageEvent:
    """Build a Hermes MessageEvent from an unwrapped NIP-17 rumor."""
    created_at = rumor_data.get("created_at")
    timestamp = datetime.fromtimestamp(created_at) if isinstance(created_at, int) else datetime.now()

    source = SessionSource(
        platform=Platform.NOSTR,
        chat_id=sender_hex,
        chat_name=sender_display,
        chat_type="dm",
        user_id=sender_hex,
        user_name=sender_display,
    )
    return MessageEvent(
        text=rumor_data.get("content", ""),
        message_type=MessageType.TEXT,
        source=source,
        raw_message=raw_message,
        message_id=rumor_data.get("id") or raw_message.get("id"),
        timestamp=timestamp,
    )


def _extract_dm_relays_from_events(events: Any) -> List[str]:
    """Extract `relay` tags from the latest kind-10050 event."""
    event_data_list = [event_data for _event, event_data in _iter_valid_event_pairs(events)]
    if not event_data_list:
        return []

    latest_created_at = max(_event_created_at(event_data) for event_data in event_data_list)
    latest_event_data = next(
        event_data
        for event_data in reversed(event_data_list)
        if _event_created_at(event_data) == latest_created_at
    )

    relays: List[str] = []
    seen: set[str] = set()
    for tag in latest_event_data.get("tags", []):
        if not isinstance(tag, list) or len(tag) < 2 or tag[0] != "relay":
            continue
        relay = str(tag[1]).strip()
        if relay and relay not in seen:
            seen.add(relay)
            relays.append(relay)
    return relays


def _build_kind(sdk: Any, value: int) -> Any:
    kind_cls = getattr(sdk, "Kind")
    return kind_cls(value)


def _filter_for_recipient_gift_wraps(sdk: Any, recipient: Any) -> Any:
    flt = sdk.Filter()
    flt = flt.kind(_build_kind(sdk, GIFT_WRAP_KIND))
    if hasattr(flt, "pubkey"):
        flt = flt.pubkey(recipient)
    elif hasattr(flt, "pubkeys"):
        flt = flt.pubkeys([recipient])
    flt = flt.limit(FETCH_LIMIT)
    return flt


def _filter_for_dm_relays(sdk: Any, author: Any) -> Any:
    flt = sdk.Filter()
    flt = flt.kind(_build_kind(sdk, DM_RELAY_LIST_KIND))
    if hasattr(flt, "author"):
        flt = flt.author(author)
    elif hasattr(flt, "authors"):
        flt = flt.authors([author])
    flt = flt.limit(1)
    return flt


def _build_dm_relay_list_builder(sdk: Any, relays: List[str]) -> Any:
    """Build a kind-10050 event advertising this identity's inbox relays."""
    tags = [sdk.Tag.parse(["relay", relay]) for relay in _parse_relay_list(relays)]
    return sdk.EventBuilder(_build_kind(sdk, DM_RELAY_LIST_KIND), "").tags(tags)


async def _setup_runtime_with_keys(secret_key: str, relays: List[str]) -> tuple[Any, Any, Any, Any, str]:
    """Create signer/client/public-key state for Nostr operations."""
    sdk = _load_nostr_sdk()
    maybe_set_loop = getattr(sdk, "uniffi_set_event_loop", None)
    if callable(maybe_set_loop):
        maybe_set_loop(asyncio.get_running_loop())

    keys = sdk.Keys.parse(secret_key)
    signer = sdk.NostrSigner.keys(keys)
    client = sdk.Client(signer)
    for relay in relays:
        relay_url = sdk.RelayUrl.parse(relay)
        await client.add_relay(relay_url)
        add_discovery = getattr(client, "add_discovery_relay", None)
        if callable(add_discovery):
            await add_discovery(relay_url)
    await client.connect()
    public_key = keys.public_key()
    return sdk, signer, client, public_key, _public_key_to_hex(public_key)


async def send_nostr_dm_once(secret_key: str, relays: List[str], chat_id: str, content: str) -> SendResult:
    """Send a NIP-17 DM without requiring a long-lived adapter instance."""
    sdk = client = None
    relays = _parse_relay_list(relays)
    if not secret_key:
        return SendResult(success=False, error="NOSTR_SECRET_KEY is required")
    if not relays:
        return SendResult(success=False, error="NOSTR_RELAYS must contain at least one relay")

    try:
        sdk, _signer, client, _public_key, _public_key_hex = await _setup_runtime_with_keys(secret_key, relays)
        recipient = sdk.PublicKey.parse(chat_id)
        relay_filter = _filter_for_dm_relays(sdk, recipient)
        relay_events = await client.fetch_events(relay_filter, FETCH_TIMEOUT)
        dm_relays = _extract_dm_relays_from_events(relay_events)
        if not dm_relays:
            return SendResult(
                success=False,
                error="Recipient has no kind 10050 DM relay list published",
            )

        for relay in dm_relays:
            relay_url = sdk.RelayUrl.parse(relay)
            add_write = getattr(client, "add_write_relay", None)
            if callable(add_write):
                await add_write(relay_url)
            else:
                await client.add_relay(relay_url)
        await client.connect()
        await client.send_private_msg(recipient, content)
        return SendResult(success=True)
    except Exception as exc:
        logger.warning("Nostr: failed to send DM to %s: %s", chat_id, exc)
        return SendResult(success=False, error=str(exc))
    finally:
        if client is not None:
            try:
                await client.disconnect()
            except Exception:
                pass


class NostrAdapter(BasePlatformAdapter):
    """Hermes gateway adapter for Nostr NIP-17 direct messages."""

    platform = Platform.NOSTR
    MAX_MESSAGE_LENGTH = 4000

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.NOSTR)
        self.secret_key = config.token or (config.extra or {}).get("secret_key", "")
        self.relays = _parse_relay_list((config.extra or {}).get("relays", []))
        self._sdk = None
        self._client = None
        self._signer = None
        self._public_key = None
        self._public_key_hex = ""
        self._public_key_bech32 = ""
        self._poll_task: Optional[asyncio.Task] = None
        self._seen_event_ids: set[str] = set()
        self._token_lock_identity: Optional[str] = None

    async def connect(self) -> bool:
        """Connect to configured relays and start the inbound poll loop."""
        if not self.secret_key:
            logger.error("Nostr: NOSTR_SECRET_KEY is required")
            return False
        if not self.relays:
            logger.error("Nostr: NOSTR_RELAYS must contain at least one relay")
            return False

        try:
            self._sdk, self._signer, self._client, self._public_key, self._public_key_hex = await _setup_runtime_with_keys(
                self.secret_key,
                self.relays,
            )
            self._public_key_bech32 = _public_key_to_bech32(self._public_key)
        except Exception as exc:
            logger.error("Nostr: failed to initialize client: %s", exc)
            return False

        try:
            from gateway.status import acquire_scoped_lock

            self._token_lock_identity = self._public_key_hex
            acquired, existing = acquire_scoped_lock(
                "nostr-pubkey",
                self._token_lock_identity,
                metadata={"platform": self.platform.value},
            )
            if not acquired:
                owner_pid = existing.get("pid") if isinstance(existing, dict) else None
                message = (
                    "Another local Hermes gateway is already using this Nostr identity"
                    + (f" (PID {owner_pid})." if owner_pid else ".")
                    + " Stop the other gateway before starting a second Nostr listener."
                )
                self._set_fatal_error("nostr_pubkey_lock", message, retryable=False)
                await self._safe_disconnect_client()
                return False
        except Exception as exc:
            logger.warning("Nostr: Could not acquire pubkey lock (non-fatal): %s", exc)

        try:
            await self._prime_seen_events()
        except Exception as exc:
            logger.warning("Nostr: failed to prime seen event cache: %s", exc)

        try:
            await self._publish_dm_relays()
        except Exception as exc:
            logger.warning("Nostr: failed to publish kind 10050 DM relays: %s", exc)

        self._poll_task = asyncio.create_task(self._poll_inbox())
        self._mark_connected()
        logger.info("Nostr: connected as %s via %d relay(s)", self._public_key_bech32, len(self.relays))
        return True

    async def disconnect(self) -> None:
        """Stop the inbox poll loop and close relay connections."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        await self._safe_disconnect_client()

        if self._token_lock_identity:
            try:
                from gateway.status import release_scoped_lock
                release_scoped_lock("nostr-pubkey", self._token_lock_identity)
            except Exception as exc:
                logger.warning("Nostr: Error releasing pubkey lock: %s", exc, exc_info=True)
            self._token_lock_identity = None

        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a NIP-17 direct message to a recipient pubkey."""
        if not self._client or not self._sdk:
            return SendResult(success=False, error="Nostr adapter is not connected")

        try:
            recipient = self._sdk.PublicKey.parse(chat_id)
            dm_relays = await self._fetch_dm_relays(recipient)
            if not dm_relays:
                return SendResult(
                    success=False,
                    error="Recipient has no kind 10050 DM relay list published",
                )

            for relay in dm_relays:
                relay_url = self._sdk.RelayUrl.parse(relay)
                add_write = getattr(self._client, "add_write_relay", None)
                if callable(add_write):
                    await add_write(relay_url)
                else:
                    await self._client.add_relay(relay_url)
            await self._client.connect()
            await self._client.send_private_msg(recipient, content)
            return SendResult(success=True)
        except Exception as exc:
            logger.warning("Nostr: failed to send DM to %s: %s", chat_id, exc)
            return SendResult(success=False, error=str(exc))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Nostr has no standardized typing indicator for this adapter."""
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic Nostr chat information."""
        return {"name": chat_id, "type": "dm"}

    async def _safe_disconnect_client(self) -> None:
        client = self._client
        self._client = None
        self._signer = None
        self._sdk = None
        if client is None:
            return
        try:
            await client.disconnect()
        except Exception:
            pass

    async def _prime_seen_events(self) -> None:
        """Seed the seen-event cache so startup doesn't replay historical DMs."""
        if not self._client or not self._sdk or not self._public_key:
            return
        flt = _filter_for_recipient_gift_wraps(self._sdk, self._public_key)
        events = await self._client.fetch_events(flt, FETCH_TIMEOUT)
        for _event, event_data in _iter_valid_event_pairs(events):
            event_id = event_data.get("id")
            if event_id:
                self._remember_event_id(event_id)

    async def _poll_inbox(self) -> None:
        """Poll relays for new gift wraps addressed to this identity."""
        while True:
            try:
                await self._poll_inbox_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Nostr: inbox poll failed: %s", exc)
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _poll_inbox_once(self) -> None:
        if not self._client or not self._sdk or not self._public_key:
            return

        flt = _filter_for_recipient_gift_wraps(self._sdk, self._public_key)
        events = await self._client.fetch_events(flt, FETCH_TIMEOUT)
        sorted_events = sorted(
            _iter_valid_event_pairs(events),
            key=lambda pair: _event_created_at(pair[1]),
        )

        for event, event_data in sorted_events:
            event_id = event_data.get("id")
            if not event_id or event_id in self._seen_event_ids:
                continue
            self._remember_event_id(event_id)
            await self._handle_gift_wrap(event, event_data)

    async def _handle_gift_wrap(self, event: Any, event_data: Dict[str, Any]) -> None:
        try:
            unwrapped = await self._sdk.UnwrappedGift.from_gift_wrap(self._signer, event)
        except Exception as exc:
            logger.debug("Nostr: failed to unwrap gift wrap %s: %s", event_data.get("id"), exc)
            return

        rumor = unwrapped.rumor()
        rumor_data = _event_to_dict(rumor)
        if rumor_data.get("kind") != NIP17_DM_KIND:
            return

        sender = unwrapped.sender()
        sender_hex = _public_key_to_hex(sender)
        if sender_hex == self._public_key_hex:
            return

        message_event = _message_event_from_rumor(
            sender_hex=sender_hex,
            sender_display=_public_key_to_bech32(sender),
            rumor_data=rumor_data,
            raw_message=event_data,
        )
        await self.handle_message(message_event)

    async def _fetch_dm_relays(self, recipient: Any) -> List[str]:
        """Fetch recipient kind-10050 DM relays from the configured relay set."""
        if not self._client or not self._sdk:
            return []
        flt = _filter_for_dm_relays(self._sdk, recipient)
        events = await self._client.fetch_events(flt, FETCH_TIMEOUT)
        return _extract_dm_relays_from_events(events)

    async def _publish_dm_relays(self) -> None:
        """Publish our kind-10050 DM relay list to configured relays."""
        if not self._client or not self._sdk or not self.relays:
            return

        builder = _build_dm_relay_list_builder(self._sdk, self.relays)
        relay_urls = [self._sdk.RelayUrl.parse(relay) for relay in self.relays]
        await self._client.send_event_builder_to(relay_urls, builder)

    def _remember_event_id(self, event_id: str) -> None:
        self._seen_event_ids.add(event_id)
        if len(self._seen_event_ids) > MAX_SEEN_EVENT_IDS:
            self._seen_event_ids.pop()

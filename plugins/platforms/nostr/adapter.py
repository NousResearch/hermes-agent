"""Nostr platform adapter for Hermes Agent — plugin.

Encryption uses the modern NIP-44 (v2) scheme combined with NIP-17
gift-wrapped direct messages (kind 1059), replacing the legacy NIP-04
(kind 4) scheme.
"""

import asyncio
import hashlib
import json
import logging
import os
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional

from nostr_sdk import (
    Kind,
    Keys,
    Filter,
    Tag,
    PublicKey,
    EventBuilder,
    Client,
    NostrSigner,
    HandleNotification,
    gift_wrap,
)

from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.platforms.base import Platform
from gateway.session import SessionSource

logger = logging.getLogger(__name__)


DEFAULT_RELAYS = [
    "wss://relay.damus.io",
    "wss://relay.primal.net",
    "wss://relay.snort.social",
]

# Bounded cache of already-processed event ids, so a relay resend or a reconnect
# replay of the same event (kind 44 or 1059) is not decrypted/dispatched twice.
MAX_SEEN_EVENTS = 1000


class _NotificationHandler(HandleNotification):
    """Nostr SDK notification handler bridging sync callbacks to the adapter.

    nostr_sdk 0.44+ invokes ``handle``/``handle_msg`` synchronously from the
    notification loop for every relay message; ``handle`` receives each
    ``Event`` and defers async processing to the adapter's event loop via a
    task, so a slow handler never blocks the SDK loop.
    """

    def __init__(self, adapter: "NostrAdapter"):
        self._adapter = adapter

    def handle_msg(self, relay_url, msg):  # noqa: ARG002 - SDK callback signature
        # Raw relay messages are not events; we only care about events.
        return None

    def handle(self, relay_url, subscription_id, event):  # noqa: ARG002
        adapter = self._adapter
        if not adapter._listening:
            return
        # Defer to the async event loop; failures are contained in _process_event.
        asyncio.create_task(adapter._process_event(event))


class NostrAdapter(BasePlatformAdapter):
    """Nostr platform adapter."""

    def __init__(self, config):
        super().__init__(config, Platform("nostr"))
        extra = getattr(config, "extra", {}) or {}

        self.relays = os.getenv("NOSTR_RELAYS", "").split(",") if os.getenv("NOSTR_RELAYS") else extra.get("relays", DEFAULT_RELAYS)
        if isinstance(self.relays, str):
            self.relays = [r.strip() for r in self.relays.split(",") if r.strip()]

        self.nsec = os.getenv("NOSTR_NSEC") or extra.get("nsec", "")

        self.client: Optional[Client] = None
        self.keys: Optional[Keys] = None
        self.signer: Optional[NostrSigner] = None
        self.pubkey: Optional[str] = None
        self._listening = False
        self._lock_key: Optional[str] = None
        self._seen_event_ids: "OrderedDict[str, bool]" = OrderedDict()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self.nsec:
            logger.error("Nostr private key (nsec) not configured")
            self._set_fatal_error("config_missing", "NOSTR_NSEC must be set", retryable=False)
            return False

        # Prevent two profiles from using the same nsec
        try:
            from gateway.status import acquire_scoped_lock, release_scoped_lock
            nsec_hash = hashlib.sha256(self.nsec.encode()).hexdigest()[:16]
            self._lock_key = f"nostr:{nsec_hash}"
            if not acquire_scoped_lock("nostr", self._lock_key):
                logger.error("Nostr: this nsec is already in use by another profile")
                self._set_fatal_error("lock_conflict", "Nostr identity in use by another profile", retryable=False)
                return False
        except ImportError:
            self._lock_key = None

        try:
            self.keys = Keys.parse(self.nsec)
            self.pubkey = self.keys.public_key().to_hex()

            signer = NostrSigner.keys(self.keys)
            self.signer = signer
            self.client = Client(signer)

            for relay in self.relays:
                self.client.add_relay(relay)

            await self.client.connect()

            self._listening = True
            asyncio.create_task(self._listen_for_messages())

            logger.info("Connected to Nostr relays: %s", self.relays)
            return True

        except Exception:
            # Do NOT interpolate the exception string here. If Keys.parse(self.nsec)
            # raised, str(e) could embed the raw key material. Log a generic message
            # and let logger.exception surface the exception detail in the traceback;
            # surface only a redacted fatal error.
            logger.exception("Failed to connect to Nostr")
            if self._lock_key:
                try:
                    from gateway.status import release_scoped_lock
                    release_scoped_lock("nostr", self._lock_key)
                except Exception:
                    pass
                self._lock_key = None
            self._set_fatal_error("connect_failed", "Failed to connect to Nostr relays", retryable=True)
            return False

    async def disconnect(self):
        self._listening = False
        if self._lock_key:
            try:
                from gateway.status import release_scoped_lock
                release_scoped_lock("nostr", self._lock_key)
            except Exception:
                pass
            self._lock_key = None
        if self.client:
            await self.client.disconnect()
            self.client = None
        self.keys = None
        self.signer = None
        self.nsec = None
        self.pubkey = None
        logger.info("Disconnected from Nostr relays")

    async def _listen_for_messages(self):
        if not self.client:
            return

        # NIP-17 gift-wrapped DMs (kind 1059) plus NIP-44 direct DMs (kind 44).
        filter_obj = Filter().kinds([Kind(1059), Kind(44)])
        try:
            await self.client.subscribe(filter_obj)
        except Exception as e:
            logger.warning("Nostr subscribe failed: %s", e)

        handler = _NotificationHandler(self)
        try:
            await self.client.handle_notifications(handler)
        except Exception as e:
            logger.exception("Nostr notification handler terminated: %s", e)

    def _mark_seen(self, event_id_hex: str) -> bool:
        """Record an event id as seen.

        Returns True the first time an id is seen (and records it); False if the
        id was already processed. The cache is bounded: the oldest entries are
        evicted once it exceeds MAX_SEEN_EVENTS.
        """
        if event_id_hex in self._seen_event_ids:
            return False
        self._seen_event_ids[event_id_hex] = True
        while len(self._seen_event_ids) > MAX_SEEN_EVENTS:
            self._seen_event_ids.popitem(last=False)
        return True

    async def _process_event(self, event):
        try:
            kind = event.kind().as_u16()

            # Dedup: skip events already processed (relay resend / reconnect replay).
            try:
                event_id_hex = event.id().to_hex()
            except Exception:
                event_id_hex = None
            if event_id_hex is not None and not self._mark_seen(event_id_hex):
                logger.debug("Skipping already-processed Nostr event %s", event_id_hex)
                return

            if kind == 1059:
                await self._handle_gift_wrap(event)
            elif kind == 44:
                await self._handle_nip44_dm(event)
            else:
                logger.debug("Ignoring unsupported Nostr event kind %s", kind)
        except Exception as e:
            logger.exception("Error in _process_event: %s", e)

    async def _handle_gift_wrap(self, event):
        """Unwrap a NIP-17 gift wrap (kind 1059) and decrypt its NIP-44 payload."""
        if not self.client or not self.signer:
            logger.warning("Nostr: cannot unwrap gift wrap without client/signer")
            return
        try:
            unwrapped = await self.client.unwrap_gift_wrap(event)
            sender_pk = unwrapped.sender()          # real author of the wrapped DM
            rumor = unwrapped.rumor()                # kind-14 rumor carrying the payload
            ciphertext = rumor.content()
            plaintext = await self.signer.nip44_decrypt(sender_pk, ciphertext)
            sender_hex = sender_pk.to_hex()
            event_id = rumor.id().to_hex()
            created_at = rumor.created_at().as_secs()
            await self._handle_incoming_message(sender_hex, plaintext, event_id, created_at)
        except Exception as e:
            event_id = None
            try:
                event_id = event.id().to_hex()
            except Exception:
                pass
            logger.warning("Failed to unwrap/decrypt Nostr gift wrap %s: %s",
                           event_id or "unknown", e)

    async def _handle_nip44_dm(self, event):
        """Decrypt a direct NIP-44 DM (kind 44)."""
        if not self.signer:
            logger.warning("Nostr: cannot decrypt NIP-44 DM without signer")
            return
        try:
            sender_pk = event.author()
            plaintext = await self.signer.nip44_decrypt(sender_pk, event.content())
            sender_hex = sender_pk.to_hex()
            event_id = event.id().to_hex()
            created_at = event.created_at().as_secs()
            await self._handle_incoming_message(sender_hex, plaintext, event_id, created_at)
        except Exception as e:
            logger.warning("Failed to decrypt Nostr NIP-44 DM %s: %s",
                           event.id().to_hex(), e)

    async def _handle_incoming_message(self, sender_pubkey: str, content: str, event_id: str, timestamp: int):
        try:
            dt = datetime.fromtimestamp(timestamp)
            source = SessionSource(
                platform=Platform("nostr"),
                chat_id=sender_pubkey,
                user_id=sender_pubkey,
                user_name=sender_pubkey[:8] + "...",
            )
            event = MessageEvent(
                source=source,
                text=content,
                timestamp=dt,
            )
            await self.handle_message(event)
        except Exception as e:
            logger.exception("Error handling incoming Nostr message: %s", e)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        if not self.client or not self.keys or not self.signer:
            return SendResult(success=False, error="Not connected to Nostr relays")
        try:
            recipient_pk = PublicKey.parse(chat_id)
            # NIP-44 encrypt the DM payload to the recipient.
            ciphertext = await self.signer.nip44_encrypt(recipient_pk, content)
            # Build the inner kind-14 rumor (NIP-17).
            rumor = EventBuilder.private_msg_rumor(recipient_pk, ciphertext).build(self.keys.public_key())
            # Gift-wrap the rumor into a kind-1059 event (NIP-17/NIP-59).
            wrapped = await gift_wrap(
                self.signer,
                recipient_pk,
                rumor,
                [Tag.public_key(recipient_pk)],
            )
            output = await self.client.send_event(wrapped)
            return SendResult(success=True, message_id=output.id.to_hex())
        except Exception as e:
            logger.exception("Failed to send Nostr message: %s", e)
            return SendResult(success=False, error=f"Failed to send message: {str(e)}")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        pass

    async def send_image(self, chat_id: str, image_url: str, caption: str = "", reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        text = f"{caption}\n{image_url}" if caption else image_url
        return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

    async def get_chat_info(self, chat_id: str) -> dict:
        if not self.client:
            return {"name": chat_id, "type": "user", "chat_id": chat_id}
        try:
            from datetime import timedelta
            filter_obj = Filter().author(PublicKey.parse(chat_id)).kind(Kind(0))
            events = await self.client.fetch_events(filter_obj, timedelta(seconds=5))
            event_list = events.to_vec()
            if event_list:
                event = event_list[0]
                profile = json.loads(event.content())
                name = profile.get("display_name", profile.get("name", chat_id))
                return {"name": name, "type": "user", "chat_id": chat_id, "profile": profile}
            else:
                return {"name": chat_id, "type": "user", "chat_id": chat_id}
        except Exception as e:
            logger.warning("Failed to fetch profile for %s: %s", chat_id, e)
            return {"name": chat_id, "type": "user", "chat_id": chat_id}


# ── Plugin contract functions ────────────────────────────────────────────────

def check_nostr_requirements() -> bool:
    try:
        import nostr_sdk
        return True
    except ImportError:
        return False


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    nsec = os.getenv("NOSTR_NSEC") or extra.get("nsec", "")
    return bool(nsec)


def is_connected(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    nsec = os.getenv("NOSTR_NSEC") or extra.get("nsec", "")
    return bool(nsec)


def interactive_setup() -> None:
    from hermes_cli.setup import (
        prompt,
        prompt_yes_no,
        save_env_value,
        get_env_value,
        print_header,
        print_info,
        print_warning,
        print_success,
    )

    print_header("Nostr")
    existing_nsec = get_env_value("NOSTR_NSEC")
    if existing_nsec:
        print_info("Nostr: already configured")
        if not prompt_yes_no("Reconfigure Nostr?", False):
            return

    print_info("Connect Hermes to the Nostr protocol (NIP-44 / NIP-17 gift-wrapped DMs).")
    print_info("   Requires a Nostr nsec private key and relay URLs.")
    print()

    nsec = prompt("Nostr nsec private key", default=existing_nsec or "", password=True)
    if not nsec:
        print_warning("nsec is required — skipping Nostr setup")
        return
    save_env_value("NOSTR_NSEC", nsec.strip())

    print()
    default_relays = ",".join(DEFAULT_RELAYS)
    relays = prompt(
        "Relay URLs (comma-separated)",
        default=get_env_value("NOSTR_RELAYS") or default_relays,
    )
    if relays:
        save_env_value("NOSTR_RELAYS", relays.strip())

    print()
    print_info("📬 Home channel for cron / notification delivery")
    print_info("   This is the pubkey hex that receives cron job output.")
    home = prompt(
        "Home channel pubkey (or empty to skip)",
        default=get_env_value("NOSTR_HOME_CHANNEL") or "",
    )
    if home:
        save_env_value("NOSTR_HOME_CHANNEL", home.strip())

    print()
    print_success("Nostr configuration saved to ~/.hermes/.env")
    print_info("Restart the gateway for changes to take effect: hermes gateway restart")


def _env_enablement() -> dict | None:
    nsec = os.getenv("NOSTR_NSEC", "").strip()
    if not nsec:
        return None
    seed: dict = {
        "nsec": nsec,
    }
    relays = os.getenv("NOSTR_RELAYS", "").strip()
    if relays:
        seed["relays"] = [r.strip() for r in relays.split(",") if r.strip()]

    home = os.getenv("NOSTR_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("NOSTR_HOME_CHANNEL_NAME", home),
        }
    return seed


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Open an ephemeral Nostr connection, send a NIP-44 + NIP-17 DM, disconnect.

    Used by ``hermes cron`` when the gateway runner is not in the same
    process.  Without this hook, ``deliver=nostr`` cron jobs fail with
    ``No live adapter for platform``.
    """
    import os
    from nostr_sdk import Keys as StandaloneKeys, Client as StandaloneClient, NostrSigner as StandaloneSigner
    from nostr_sdk import PublicKey as StandalonePublicKey, Tag as StandaloneTag
    from nostr_sdk import EventBuilder as StandaloneEventBuilder, gift_wrap as standalone_gift_wrap

    extra = getattr(pconfig, "extra", {}) or {}
    nsec = os.getenv("NOSTR_NSEC") or extra.get("nsec", "")
    if not nsec:
        return {"error": "Nostr standalone send: NOSTR_NSEC must be configured"}

    relays = os.getenv("NOSTR_RELAYS", "") or extra.get("relays", DEFAULT_RELAYS)
    if isinstance(relays, str):
        relay_list = [r.strip() for r in relays.split(",") if r.strip()]
    else:
        relay_list = list(relays)

    if not relay_list:
        return {"error": "Nostr standalone send: no relays configured"}

    try:
        keys = StandaloneKeys.parse(nsec)
        signer = StandaloneSigner.keys(keys)
        client = StandaloneClient(signer)

        for relay in relay_list:
            client.add_relay(relay)

        await client.connect()

        recipient_pk = StandalonePublicKey.parse(chat_id)
        # NIP-44 encrypt the DM payload to the recipient.
        ciphertext = await signer.nip44_encrypt(recipient_pk, message)
        # Build the inner kind-14 rumor (NIP-17).
        rumor = StandaloneEventBuilder.private_msg_rumor(recipient_pk, ciphertext).build(keys.public_key())
        # Gift-wrap the rumor into a kind-1059 event (NIP-17/NIP-59).
        wrapped = await standalone_gift_wrap(
            signer,
            recipient_pk,
            rumor,
            [StandaloneTag.public_key(recipient_pk)],
        )
        output = await client.send_event(wrapped)
        await client.disconnect()

        return {"success": True, "message_id": output.id.to_hex()}

    except Exception as e:
        return {"error": f"Nostr standalone send failed: {e}"}


def register(ctx):
    ctx.register_platform(
        name="nostr",
        label="Nostr",
        adapter_factory=lambda cfg: NostrAdapter(cfg),
        check_fn=check_nostr_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["NOSTR_NSEC"],
        install_hint="pip install nostr-sdk",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="NOSTR_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        emoji="📯",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via Nostr. Messages are sent as NIP-44 encrypted "
            "gift-wrapped direct messages (NIP-17). Keep responses concise."
        ),
    )

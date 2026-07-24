"""Nostr platform adapter for Hermes Agent — plugin."""

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from nostr_sdk import Kind, Keys, Filter, Tag, EventBuilder, Client, NostrSigner

from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.platforms.base import Platform
from gateway.session import SessionSource

logger = logging.getLogger(__name__)


DEFAULT_RELAYS = [
    "wss://relay.damus.io",
    "wss://relay.primal.net",
    "wss://relay.snort.social",
]


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
        self.pubkey: Optional[str] = None
        self._listening = False
        self._lock_key: Optional[str] = None

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
            self.keys = Keys.from_nsec(self.nsec)
            self.pubkey = self.keys.public_key().to_hex()

            signer = NostrSigner.keys(self.keys)
            self.client = Client(signer)

            for relay in self.relays:
                self.client.add_relay(relay)

            await self.client.connect()

            self._listening = True
            asyncio.create_task(self._listen_for_messages())

            logger.info("Connected to Nostr relays: %s", self.relays)
            return True

        except Exception as e:
            logger.exception("Failed to connect to Nostr: %s", e)
            if self._lock_key:
                try:
                    from gateway.status import release_scoped_lock
                    release_scoped_lock("nostr", self._lock_key)
                except Exception:
                    pass
                self._lock_key = None
            self._set_fatal_error("connect_failed", str(e), retryable=True)
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
        self.nsec = None
        self.pubkey = None
        logger.info("Disconnected from Nostr relays")

    async def _listen_for_messages(self):
        if not self.client:
            return

        filter_obj = Filter().kind([4, 1])

        async def message_handler(message):
            if not self._listening:
                return
            try:
                event_dict = message.as_json_dict()
                await self._process_event(event_dict)
            except Exception:
                pass

        await self.client.handle_notifications(message_handler)

    async def _process_event(self, event_dict: dict):
        try:
            event_id = event_dict.get("id")
            pubkey = event_dict.get("pubkey")
            kind = event_dict.get("kind")
            content = event_dict.get("content")
            tags = event_dict.get("tags", [])
            created_at = event_dict.get("created_at")

            if kind == 4:
                if not self.keys:
                    return
                try:
                    decrypted = self.keys.decrypt(content, pubkey)
                    await self._handle_incoming_message(pubkey, decrypted, event_id, created_at)
                except Exception as e:
                    logger.warning("Failed to decrypt Nostr event %s: %s", event_id, e)
            elif kind == 1:
                our_pubkey_tag = [t for t in tags if t[0] == 'p' and t[1] == self.pubkey]
                if our_pubkey_tag:
                    await self._handle_incoming_message(pubkey, content, event_id, created_at)
        except Exception as e:
            logger.exception("Error in _process_event: %s", e)

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
        if not self.client or not self.keys:
            return SendResult(success=False, error="Not connected to Nostr relays")
        try:
            ciphertext = self.keys.encrypt(chat_id, content)
            event_builder = EventBuilder(Kind.EncryptedDirectMessage, ciphertext)
            event_builder.tag(Tag.pubkey(chat_id))
            signed_event = event_builder.sign_with_keys(self.keys)
            await self.client.send_event(signed_event)
            return SendResult(success=True, message_id=signed_event.id().to_hex())
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
            filter_obj = Filter().author(chat_id).kind(0)
            events = await self.client.query(filter_obj, timeout=5)
            if events:
                event = events[0]
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

    print_info("Connect Hermes to the Nostr protocol (NIP-04 encrypted DMs).")
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
    """Open an ephemeral Nostr connection, send a NIP-04 DM, and disconnect.

    Used by ``hermes cron`` when the gateway runner is not in the same
    process.  Without this hook, ``deliver=nostr`` cron jobs fail with
    ``No live adapter for platform``.
    """
    import os
    from nostr_sdk import Keys as StandaloneKeys, Client as StandaloneClient, NostrSigner as StandaloneSigner

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
        keys = StandaloneKeys.from_nsec(nsec)
        signer = StandaloneSigner.keys(keys)
        client = StandaloneClient(signer)

        for relay in relay_list:
            client.add_relay(relay)

        await client.connect()

        ciphertext = keys.encrypt(chat_id, message)
        event_builder = EventBuilder(Kind.EncryptedDirectMessage, ciphertext)
        event_builder.tag(Tag.pubkey(chat_id))
        signed_event = event_builder.sign_with_keys(keys)
        await client.send_event(signed_event)
        await client.disconnect()

        return {"success": True, "message_id": signed_event.id().to_hex()}

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
            "You are chatting via Nostr. Messages are sent as encrypted "
            "direct messages (NIP-04). Keep responses concise."
        ),
    )

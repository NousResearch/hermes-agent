"""Nostr platform adapter for Hermes Agent — plugin."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from nostr_sdk import Kind, Keys, Message, Filter, Tag, EventBuilder, Client, NostrSigner

from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.platforms.base import Platform
from gateway.session import SessionSource

logger = logging.getLogger(__name__)


class NostrAdapter(BasePlatformAdapter):
    """Nostr platform adapter."""

    def __init__(self, config):
        super().__init__(config, Platform("nostr"))
        self.relays = []
        self.client: Optional[Client] = None
        self.keys: Optional[Keys] = None
        self.nsec: Optional[str] = None
        self.pubkey: Optional[str] = None
        self._listening = False

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to Nostr relays."""
        try:
            relays = self.config.extra.get("relays", [])
            if not relays:
                import os
                relays_str = os.getenv("NOSTR_RELAYS", "")
                if relays_str:
                    relays = [r.strip() for r in relays_str.split(",") if r.strip()]
                else:
                    relays = [
                        "wss://relay.damus.io",
                        "wss://relay.primal.net",
                        "wss://relay.snort.social",
                    ]
            self.relays = relays

            nsec = self.config.extra.get("nsec")
            if not nsec:
                import os
                nsec = os.getenv("NOSTR_NSEC")
            if not nsec:
                logger.error("Nostr private key (nsec) not configured")
                return False

            self.nsec = nsec
            self.keys = Keys.from_nsec(nsec)
            self.pubkey = self.keys.public_key().to_hex()

            signer = NostrSigner.keys(self.keys)
            self.client = Client(signer)

            for relay in self.relays:
                self.client.add_relay(relay)

            await self.client.connect()

            self._listening = True
            asyncio.create_task(self._listen_for_messages())

            logger.info(f"Connected to Nostr relays: {self.relays}")
            return True

        except Exception as e:
            logger.exception(f"Failed to connect to Nostr: {e}")
            return False

    async def disconnect(self):
        self._listening = False
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
                    logger.warning(f"Failed to decrypt Nostr event {event_id}: {e}")
            elif kind == 1:
                our_pubkey_tag = [t for t in tags if t[0] == 'p' and t[1] == self.pubkey]
                if our_pubkey_tag:
                    await self._handle_incoming_message(pubkey, content, event_id, created_at)

        except Exception as e:
            logger.exception(f"Error in _process_event: {e}")

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
            logger.exception(f"Error handling incoming Nostr message: {e}")

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
            logger.exception(f"Failed to send Nostr message: {e}")
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
            logger.warning(f"Failed to fetch profile for {chat_id}: {e}")
            return {"name": chat_id, "type": "user", "chat_id": chat_id}


def check_nostr_requirements() -> bool:
    try:
        import nostr_sdk
        return True
    except ImportError:
        return False


def register(ctx):
    ctx.register_platform(
        name="nostr",
        label="Nostr",
        adapter_factory=lambda cfg: NostrAdapter(cfg),
        check_fn=check_nostr_requirements,
        required_env=["NOSTR_NSEC"],
        install_hint="pip install nostr-sdk",
        emoji="📯",
        platform_hint=(
            "You are chatting via Nostr. Messages are sent as encrypted "
            "direct messages (NIP-04). Keep responses concise."
        ),
    )

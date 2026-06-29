"""Nostr platform adapter for Hermes Agent."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from nostr_sdk import Kind, Keys, Message, Filter, Tag, EventBuilder, Client, NostrSigner

from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult, Platform

logger = logging.getLogger(__name__)


class NostrAdapter(BasePlatformAdapter):
    """Nostr platform adapter."""

    def __init__(self, config):
        super().__init__(config, Platform.NOSTR)
        self.relays = []  # list of relay URLs
        self.client: Optional[Client] = None
        self.keys: Optional[Keys] = None
        self.nsec: Optional[str] = None  # private key in nsec format
        self.pubkey: Optional[str] = None  # public key in hex
        self._listening = False

    async def connect(self) -> bool:
        """Connect to Nostr relays."""
        try:
            # Get configuration
            relays = self.config.extra.get("relays", [])
            if not relays:
                # Try to get from environment variables
                import os
                relays_str = os.getenv("NOSRT_RELAYS", "")
                if relays_str:
                    relays = [r.strip() for r in relays_str.split(",") if r.strip()]
                else:
                    # Default relays
                    relays = [
                        "wss://relay.damus.io",
                        "wss://relay.primal.net",
                        "wss://relay.snort.social",
                    ]

            self.relays = relays

            # Get private key from config or environment
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

            # Create Nostr client
            signer = NostrSigner.keys(self.keys)
            self.client = Client(signer)

            # Add relays
            for relay in self.relays:
                self.client.add_relay(relay)

            # Connect to relays
            await self.client.connect()

            # Set up listener for incoming messages
            self._listening = True
            asyncio.create_task(self._listen_for_messages())

            logger.info(f"Connected to Nostr relays: {self.relays}")
            return True

        except Exception as e:
            logger.exception(f"Failed to connect to Nostr: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Nostr relays."""
        self._listening = False
        if self.client:
            await self.client.disconnect()
            self.client = None
        self.keys = None
        self.nsec = None
        self.pubkey = None
        logger.info("Disconnected from Nostr relays")

    async def _listen_for_messages(self):
        """Listen for incoming Nostr events and dispatch them as messages."""
        if not self.client:
            return

        # Filter for kind 4 (encrypted direct messages) and kind 1 (text notes) that mention us
        filter_obj = Filter().kind([4, 1])

        async def message_handler(message):
            if not self._listening:
                return

            try:
                if message.type == MessageType.EVENT:
                    event = event.as_json_dict()
                    await self._process_event(event)
            except Exception as e:
                logger.exception(f"Error processing Nostr event: {e}")

        await self.client.handle_notifications(message_handler)

    async def _process_event(self, event_dict: dict):
        """Process an incoming Nostr event and dispatch as a message if it's for us."""
        try:
            event_id = event_dict.get("id")
            pubkey = event_dict.get("pubkey")
            kind = event_dict.get("kind")
            content = event_dict.get("content")
            tags = event_dict.get("tags", [])
            created_at = event_dict.get("created_at")

            # We only want to process events that are directed to us (our pubkey)
            # For kind 4 (encrypted direct messages), we can decrypt and see if it's for us.
            # For kind 1, we can check if there's a 'p' tag with our pubkey (mention).

            if kind == 4:
                # Encrypted direct message
                if not self.keys:
                    return
                try:
                    # Decrypt the message
                    decrypted = self.keys.decrypt(content, pubkey)
                    plaintext = decrypted
                    await self._handle_incoming_message(pubkey, plaintext, event_id, created_at)
                except Exception as e:
                    logger.warning(f"Failed to decrypt Nostr event {event_id}: {e}")
            elif kind == 1:
                # Plain text note - check if it mentions us
                # We'll look for a 'p' tag that matches our pubkey
                our_pubkey_tag = [t for t in tags if t[0] == 'p' and t[1] == self.pubkey]
                if our_pubkey_tag:
                    # This is a mention directed at us
                    await self._handle_incoming_message(pubkey, content, event_id, created_at)

        except Exception as e:
            logger.exception(f"Error in _process_event: {e}")

    async def _handle_incoming_message(self, sender_pubkey: str, content: str, event_id: str, timestamp: int):
        """Handle an incoming message and dispatch to the gateway."""
        try:
            # Convert timestamp to datetime
            dt = datetime.fromtimestamp(timestamp)

            # Create a message event
            event = MessageEvent(
                platform=Platform.NOSTR,
                channel_id=sender_pubkey,  # Use sender's pubkey as channel ID for DMs
                thread_id=None,  # Nostr doesn't have threads in the same way; we could use event ID?
                user_id=sender_pubkey,
                user_name=sender_pubkey[:8] + "...",  # Shortened pubkey for display
                message_type=MessageType.TEXT,
                text_content=content,
                timestamp=dt
            )
            # Dispatch to the gateway via self.handle_message(event)
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
        """Send a text message via Nostr."""
        if not self.client or not self.keys:
            return SendResult.error("Not connected to Nostr relays")

        try:
            # Determine if we are sending an encrypted direct message (kind 4) or a plain note (kind 1)
            # We'll treat the chat_id as the recipient's pubkey (in hex)
            recipient_pubkey = chat_id

            # Encrypt the message for the recipient
            ciphertext = self.keys.encrypt(recipient_pubkey, content)
            # Create the event
            event_builder = EventBuilder(Kind.EncryptedDirectMessage, ciphertext)
            # Add the recipient's pubkey as a 'p' tag
            event_builder.tag(Tag.pubkey(recipient_pubkey))
            # Sign and send the event
            signed_event = event_builder.sign_with_keys(self.keys)
            await self.client.send_event(signed_event)

            # Wait for the event to be stored? We'll consider it sent when the client accepts it.
            # The client's send_event returns when the event has been published to at least one relay.
            return SendResult.success(message_id=signed_event.id().to_hex())

        except Exception as e:
            logger.exception(f"Failed to send Nostr message: {e}")
            return SendResult.error(f"Failed to send message: {str(e)}")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send a typing indicator. Nostr doesn't have a native typing indicator, so we skip."""
        # Nostr doesn't have a standard for typing indicators.
        # We could send a custom event kind, but for now we do nothing.
        pass

    async def send_image(self, chat_id: str, image_url: str, caption: str = "", reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an image via Nostr. We'll send a text message with the image URL and caption."""
        text = f"{caption}\n{image_url}" if caption else image_url
        return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

    async def get_chat_info(self, chat_id: str) -> dict:
        """Get information about a chat (user)."""
        # We can try to fetch the user's profile (kind 0) from relays
        if not self.client:
            return {"name": chat_id, "type": "user", "chat_id": chat_id}

        try:
            # Fetch the metadata event (kind 0) for the user
            filter_obj = Filter().author(chat_id).kind(0)
            events = await self.client.query(filter_obj, timeout=5)
            if events:
                # Take the most recent
                event = events[0]
                profile = json.loads(event.content())
                name = profile.get("display_name", profile.get("name", chat_id))
                return {
                    "name": name,
                    "type": "user",
                    "chat_id": chat_id,
                    "profile": profile
                }
            else:
                return {"name": chat_id, "type": "user", "chat_id": chat_id}
        except Exception as e:
            logger.warning(f"Failed to fetch profile for {chat_id}: {e}")
            return {"name": chat_id, "type": "user", "chat_id": chat_id}


def check_nostr_requirements() -> bool:
    """Check if the nostr-sdk is installed."""
    try:
        import nostr_sdk
        return True
    except ImportError:
        return False
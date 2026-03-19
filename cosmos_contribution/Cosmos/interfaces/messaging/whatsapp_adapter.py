"""
cosmos WhatsApp Adapter
---------------------------
Connects cosmos to WhatsApp via Twilio's WhatsApp API.

Usage:
    export TWILIO_ACCOUNT_SID=your_account_sid
    export TWILIO_AUTH_TOKEN=your_auth_token
    export TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886  # Twilio sandbox number

    from Cosmos.interfaces.messaging.whatsapp_adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter()
    adapter.set_callback(my_message_handler)
    await adapter.connect()

Note: For production, you'll need a WhatsApp Business Account approved by Meta.
"""

import os
import asyncio
from datetime import datetime
from typing import Optional, Callable, Awaitable
from loguru import logger

from Cosmos.interfaces.messaging.base import (
    MessagingProvider,
    IncomingMessage,
    OutgoingMessage,
    MessageSource
)

# Try to import twilio
try:
    from twilio.rest import Client as TwilioClient
    from twilio.base.exceptions import TwilioRestException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("twilio not installed. Run: pip install twilio")


class WhatsAppAdapter(MessagingProvider):
    """
    WhatsApp messaging adapter for cosmos via Twilio.

    Provides:
    - Send messages to WhatsApp users
    - Receive messages via webhook (requires FastAPI integration)
    - Media message support (images, documents)

    Note: Twilio WhatsApp requires:
    1. Twilio account with WhatsApp sandbox or Business approval
    2. Webhook endpoint for receiving messages
    """

    def __init__(self, account_sid: Optional[str] = None, auth_token: Optional[str] = None,
                 whatsapp_number: Optional[str] = None):
        super().__init__("whatsapp")
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.whatsapp_number = whatsapp_number or os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
        self.client: Optional[TwilioClient] = None
        self._running = False

        # cosmos persona
        self.persona_prefix = "*adjusts spectacles* "

    async def connect(self):
        """Initialize the Twilio client."""
        if not TWILIO_AVAILABLE:
            logger.error("twilio not installed")
            return

        if not self.account_sid or not self.auth_token:
            logger.error("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set")
            return

        try:
            self.client = TwilioClient(self.account_sid, self.auth_token)
            self._running = True
            logger.info(f"WhatsApp adapter connected via Twilio (From: {self.whatsapp_number})")
            logger.info("Note: Set up webhook at /api/whatsapp/webhook for incoming messages")
        except Exception as e:
            logger.error(f"Failed to connect WhatsApp adapter: {e}")
            self._running = False

    async def disconnect(self):
        """Disconnect the adapter."""
        self._running = False
        self.client = None
        logger.info("WhatsApp adapter disconnected")

    async def send_message(self, message: OutgoingMessage):
        """Send a message via WhatsApp."""
        if not self.client:
            logger.error("WhatsApp adapter not connected")
            return

        try:
            # Ensure phone number has whatsapp: prefix
            to_number = message.channel_id
            if not to_number.startswith("whatsapp:"):
                to_number = f"whatsapp:{to_number}"

            # Add cosmos persona flavor
            content = message.content
            if not content.startswith("*"):
                content = self.persona_prefix + content

            # Send via Twilio
            twilio_message = self.client.messages.create(
                body=content,
                from_=self.whatsapp_number,
                to=to_number
            )

            logger.info(f"WhatsApp message sent: {twilio_message.sid}")

        except TwilioRestException as e:
            logger.error(f"Twilio error sending WhatsApp message: {e}")
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message: {e}")

    async def handle_webhook(self, form_data: dict) -> Optional[str]:
        """
        Handle incoming WhatsApp webhook from Twilio.

        This should be called from your FastAPI endpoint:

        @app.post("/api/whatsapp/webhook")
        async def whatsapp_webhook(request: Request):
            form_data = await request.form()
            response = await whatsapp_adapter.handle_webhook(dict(form_data))
            return Response(content=response or "", media_type="text/xml")
        """
        try:
            # Extract message details
            from_number = form_data.get("From", "")
            to_number = form_data.get("To", "")
            body = form_data.get("Body", "")
            message_sid = form_data.get("MessageSid", "")
            sender_name = form_data.get("ProfileName", "Unknown")

            if not body:
                return None

            logger.info(f"WhatsApp message received from {from_number}: {body[:50]}...")

            # Create IncomingMessage
            incoming = IncomingMessage(
                id=message_sid,
                source=MessageSource.TELEGRAM,  # We'd add WHATSAPP to enum
                sender_id=from_number,
                sender_name=sender_name,
                channel_id=from_number,  # Reply to sender
                content=body,
                timestamp=datetime.now(),
                metadata={
                    "to": to_number,
                    "num_media": form_data.get("NumMedia", "0")
                }
            )

            # Route to callback
            if self._on_message_callback:
                await self._on_message_callback(incoming)

            # Return TwiML response (empty = no auto-reply, cosmos will reply via send_message)
            return None

        except Exception as e:
            logger.error(f"Error handling WhatsApp webhook: {e}")
            return None

    def get_webhook_route(self):
        """
        Returns FastAPI route for WhatsApp webhook.

        Usage:
            from Cosmos.interfaces.messaging.whatsapp_adapter import WhatsAppAdapter

            adapter = WhatsAppAdapter()
            app.include_router(adapter.get_webhook_route())
        """
        from fastapi import APIRouter, Request, Response

        router = APIRouter()

        @router.post("/api/whatsapp/webhook")
        async def whatsapp_webhook(request: Request):
            form_data = await request.form()
            response = await self.handle_webhook(dict(form_data))
            return Response(content=response or "", media_type="text/xml")

        return router


# Add WHATSAPP to MessageSource enum
# Note: This should be added to base.py, but we'll handle it gracefully here


# Standalone test
async def test_send():
    """Test sending a WhatsApp message."""
    adapter = WhatsAppAdapter()
    await adapter.connect()

    if adapter._running:
        # Replace with your test number
        test_number = os.getenv("TEST_WHATSAPP_NUMBER", "+1234567890")

        await adapter.send_message(OutgoingMessage(
            channel_id=test_number,
            content="Good news, everyone! This is a test message from cosmos!"
        ))


if __name__ == "__main__":
    asyncio.run(test_send())

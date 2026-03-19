"""
cosmos Office 365 Integration.

"I am already in your cloud!"

Features:
1. Email: Read/Send emails via Outlook.
2. Calendar: Sync events.
3. OneDrive: Access files.
"""

import asyncio
from typing import Dict, Any, List
from loguru import logger

# Check imports
try:
    from O365 import Account
except ImportError:
    Account = None

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus

class Office365Provider(ExternalProvider):
    def __init__(self, client_id: str, client_secret: str):
        super().__init__(IntegrationConfig(name="office365"))
        self.client_id = client_id
        self.client_secret = client_secret
        self.account = None
        
    async def connect(self) -> bool:
        if not Account:
            logger.error("O365 library not installed. Run `pip install O365`")
            return False

        credentials = (self.client_id, self.client_secret)
        
        # Note: This usually requires an interactive token dance for the first time
        # We assume token backend is filesystem for now
        self.account = Account(credentials)
        
        loop = asyncio.get_event_loop()
        is_authenticated = await loop.run_in_executor(
            None, lambda: self.account.is_authenticated
        )

        if not is_authenticated:
            logger.warning("O365: Not authenticated. Interactive login required.")
            # In a real CLI app, we would trigger the login flow here
            # account.authenticate(scopes=['basic', 'message_all'])
            return False
            
        logger.info("Office 365: Connected")
        self.status = ConnectionStatus.CONNECTED
        return True

    async def sync(self):
        """Poll for new emails."""
        if self.status != ConnectionStatus.CONNECTED:
            return

        loop = asyncio.get_event_loop()
        def _get_messages():
            mailbox = self.account.mailbox()
            return list(mailbox.get_messages(limit=5))

        messages = await loop.run_in_executor(None, _get_messages)
        for msg in messages:
            await nexus.emit(
                SignalType.EXTERNAL_ALERT,
                {"provider": "o365", "type": "email", "subject": msg.subject, "from": msg.sender.address},
                source="o365_provider"
            )

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Office 365 not connected")

        loop = asyncio.get_event_loop()

        if action == "send_email":
            to = params.get('to')
            subject = params.get('subject')
            body = params.get('body')
            
            def _send():
                m = self.account.new_message()
                m.to.add(to)
                m.subject = subject
                m.body = body
                return m.send()

            await loop.run_in_executor(None, _send)
            return {"status": "sent"}
            
        elif action == "get_calendar":
            def _get_events():
                schedule = self.account.schedule()
                calendar = schedule.get_default_calendar()
                return list(calendar.get_events(limit=params.get('limit', 10)))
            
            events = await loop.run_in_executor(None, _get_events)
            return [{"id": e.object_id, "subject": e.subject, "start": e.start.isoformat()} for e in events]
            
        else:
            raise ValueError(f"Unknown action: {action}")

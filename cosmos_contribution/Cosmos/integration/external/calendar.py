"""
cosmos Calendar Integration via Google API.
"""

import asyncio
import os.path
from typing import Dict, Any, List
from datetime import datetime
from loguru import logger

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus

SCOPES = ['https://www.googleapis.com/auth/calendar']

class CalendarProvider(ExternalProvider):
    def __init__(self, creds_path: str):
        super().__init__(IntegrationConfig(name="google_calendar"))
        self.creds_path = creds_path
        self.service = None
        
    async def connect(self) -> bool:
        creds = None
        # Logic to load token.json if exists
        token_path = 'token.json'
        
        loop = asyncio.get_event_loop()
        
        try:
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)
            
            # If no valid creds, we can't do interactive auth in headless mode easily
            # This assumes token.json is provisioned or environment setup
            
            if not creds or not creds.valid:
                 logger.warning("Google Calendar: No valid token found.")
                 return False

            self.service = await loop.run_in_executor(
                None, lambda: build('calendar', 'v3', credentials=creds)
            )
            
            self.status = ConnectionStatus.CONNECTED
            return True
            
        except Exception as e:
            logger.error(f"Calendar connect error: {e}")
            return False

    async def sync(self):
        """
        Sync calendar data by refreshing cached events.

        Fetches upcoming events and stores them for quick access.
        """
        if not self.service:
            logger.debug("Calendar sync skipped: not connected")
            return

        try:
            # Refresh upcoming events cache
            events = await self.get_upcoming_events(limit=50)
            self._cached_events = events
            self._last_sync = datetime.utcnow()
            logger.info(f"Calendar synced: {len(events)} events cached")
        except Exception as e:
            logger.warning(f"Calendar sync failed: {e}")

    async def get_upcoming_events(self, limit: int = 5) -> List[Dict]:
        if not self.service: return []
        
        loop = asyncio.get_event_loop()
        now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        
        events_result = await loop.run_in_executor(
            None,
            lambda: self.service.events().list(
                calendarId='primary', timeMin=now,
                maxResults=limit, singleEvents=True,
                orderBy='startTime'
            ).execute()
        )
        return events_result.get('items', [])

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        if not self.service:
            raise ConnectionError("Google Calendar not connected")
            
        loop = asyncio.get_event_loop()
        
        if action == "schedule_event":
            event = {
                'summary': params.get('summary'),
                'location': params.get('location', ''),
                'description': params.get('description', ''),
                'start': {
                    'dateTime': params.get('start_time'), # ISO format e.g. 2026-01-25T20:00:00Z
                    'timeZone': params.get('timezone', 'UTC'),
                },
                'end': {
                    'dateTime': params.get('end_time'),
                    'timeZone': params.get('timezone', 'UTC'),
                },
                'attendees': [{'email': e} for e in params.get('attendees', [])],
            }
            
            res = await loop.run_in_executor(
                None, 
                lambda: self.service.events().insert(calendarId='primary', body=event).execute()
            )
            return {"id": res.get('id'), "htmlLink": res.get('htmlLink')}
            
        elif action == "delete_event":
            event_id = params.get('event_id')
            await loop.run_in_executor(
                None,
                lambda: self.service.events().delete(calendarId='primary', eventId=event_id).execute()
            )
            return {"status": "deleted"}

        raise ValueError(f"Unknown action: {action}")

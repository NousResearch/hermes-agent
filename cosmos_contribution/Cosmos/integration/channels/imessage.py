"""
Cosmos iMessage Channel Adapter
====================================

iMessage integration for the Cosmos swarm (macOS only).

Features:
- Direct messages
- Group chats
- Rich links
- Tapbacks (reactions)
- Attachments
- Read receipts
- Typing indicators

Uses macOS native APIs via AppleScript/JXA or sqlite3 for reading.

Setup:
1. Must run on macOS
2. Grant Full Disk Access to terminal/Python
3. Grant Accessibility permissions for sending
4. Messages app must be configured

"iMessage: Apple's walled garden. We found a door." - The Collective
"""

import os
import sys
import asyncio
import sqlite3
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import tempfile
from loguru import logger

from .channel_hub import (
    BaseChannel,
    ChannelConfig,
    ChannelMessage,
    ChannelType,
)

# Check if running on macOS
IS_MACOS = sys.platform == "darwin"


class iMessageChannel(BaseChannel):
    """
    iMessage channel adapter for macOS.

    Uses:
    - sqlite3 to read Messages database
    - AppleScript to send messages
    - Polling for new messages (no push API)

    Supports:
    - Direct messages (phone/email)
    - Group chats
    - Attachments (send and receive)
    - Tapbacks (limited)
    """

    def __init__(
        self,
        config: ChannelConfig = None,
        poll_interval: float = 2.0,
        db_path: str = None
    ):
        """
        Initialize iMessage channel.

        Args:
            config: Channel configuration
            poll_interval: Seconds between polling for new messages
            db_path: Path to chat.db (default: ~/Library/Messages/chat.db)
        """
        config = config or ChannelConfig(channel_type=ChannelType.IMESSAGE)
        super().__init__(config)

        self.poll_interval = poll_interval
        self.db_path = db_path or os.path.expanduser("~/Library/Messages/chat.db")

        self._poll_task: Optional[asyncio.Task] = None
        self._last_rowid = 0
        self._db_conn: Optional[sqlite3.Connection] = None

    async def connect(self) -> bool:
        """Connect to iMessage (start polling)."""
        if not IS_MACOS:
            logger.error("iMessage adapter only works on macOS")
            return False

        if not Path(self.db_path).exists():
            logger.error(f"Messages database not found: {self.db_path}")
            return False

        try:
            # Test database access
            self._db_conn = sqlite3.connect(self.db_path)
            cursor = self._db_conn.cursor()

            # Get latest message ID
            cursor.execute("SELECT MAX(ROWID) FROM message")
            result = cursor.fetchone()
            self._last_rowid = result[0] or 0

            logger.info(f"iMessage connected, starting from message {self._last_rowid}")

            # Start polling
            self._poll_task = asyncio.create_task(self._poll_messages())

            self._connected = True
            return True

        except sqlite3.OperationalError as e:
            logger.error(f"Cannot access Messages database: {e}")
            logger.info("Grant Full Disk Access: System Preferences > Security > Privacy > Full Disk Access")
            return False
        except Exception as e:
            logger.error(f"iMessage connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from iMessage."""
        if self._poll_task:
            self._poll_task.cancel()

        if self._db_conn:
            self._db_conn.close()

        self._connected = False
        logger.info("iMessage disconnected")

    async def _poll_messages(self):
        """Poll for new messages."""
        try:
            while True:
                await asyncio.sleep(self.poll_interval)
                await self._check_new_messages()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"iMessage poll error: {e}")

    async def _check_new_messages(self):
        """Check for new messages since last poll."""
        if not self._db_conn:
            return

        try:
            cursor = self._db_conn.cursor()

            # Query new messages
            cursor.execute("""
                SELECT
                    m.ROWID,
                    m.guid,
                    m.text,
                    m.date,
                    m.is_from_me,
                    m.handle_id,
                    h.id as sender_id,
                    c.chat_identifier,
                    c.display_name,
                    c.group_id
                FROM message m
                LEFT JOIN handle h ON m.handle_id = h.ROWID
                LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                LEFT JOIN chat c ON cmj.chat_id = c.ROWID
                WHERE m.ROWID > ?
                ORDER BY m.ROWID ASC
            """, (self._last_rowid,))

            rows = cursor.fetchall()

            for row in rows:
                rowid, guid, text, date, is_from_me, handle_id, sender_id, chat_id, display_name, group_id = row

                self._last_rowid = rowid

                # Skip own messages
                if is_from_me:
                    continue

                # Skip empty messages
                if not text:
                    continue

                is_group = group_id is not None

                # Convert macOS timestamp (seconds since 2001-01-01)
                timestamp = datetime(2001, 1, 1) + timedelta(seconds=date / 1e9)

                message = ChannelMessage(
                    message_id=guid,
                    channel_type=ChannelType.IMESSAGE,
                    channel_id=chat_id or sender_id,
                    sender_id=sender_id or "unknown",
                    sender_name=sender_id.split("@")[0] if sender_id and "@" in sender_id else sender_id,
                    text=text,
                    is_group=is_group,
                    group_name=display_name if is_group else None,
                    timestamp=timestamp,
                    raw_data={"rowid": rowid, "guid": guid}
                )

                await self._handle_inbound(message)

            # Check for attachments in new messages
            # (simplified - real implementation would join attachment table)

        except Exception as e:
            logger.error(f"Check messages error: {e}")
            # Reconnect if database locked
            try:
                self._db_conn.close()
                self._db_conn = sqlite3.connect(self.db_path)
            except Exception:
                pass

    async def send_message(
        self,
        recipient: str,
        text: str,
        media_path: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Send an iMessage.

        Args:
            recipient: Phone number or email
            text: Message text
            media_path: Optional attachment path

        Returns:
            True if sent successfully
        """
        if not IS_MACOS:
            return False

        if not self._check_rate_limit(recipient):
            logger.warning(f"Rate limit exceeded for {recipient}")
            return False

        try:
            # Escape text for AppleScript
            escaped_text = text.replace("\\", "\\\\").replace('"', '\\"')

            if media_path and Path(media_path).exists():
                # Send with attachment
                script = f'''
                    tell application "Messages"
                        set targetBuddy to "{recipient}"
                        set targetService to id of 1st service whose service type = iMessage
                        set theBuddy to buddy targetBuddy of service id targetService
                        send POSIX file "{media_path}" to theBuddy
                        if "{escaped_text}" is not "" then
                            send "{escaped_text}" to theBuddy
                        end if
                    end tell
                '''
            else:
                # Send text only
                script = f'''
                    tell application "Messages"
                        set targetBuddy to "{recipient}"
                        set targetService to id of 1st service whose service type = iMessage
                        set theBuddy to buddy targetBuddy of service id targetService
                        send "{escaped_text}" to theBuddy
                    end tell
                '''

            # Execute AppleScript
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                logger.error(f"AppleScript error: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.error("iMessage send timed out")
            return False
        except Exception as e:
            logger.error(f"iMessage send failed: {e}")
            return False

    async def send_to_group(
        self,
        group_name: str,
        text: str,
        media_path: Optional[str] = None
    ) -> bool:
        """
        Send a message to a group chat.

        Args:
            group_name: Name of the group chat
            text: Message text
            media_path: Optional attachment path

        Returns:
            True if sent successfully
        """
        if not IS_MACOS:
            return False

        try:
            escaped_text = text.replace("\\", "\\\\").replace('"', '\\"')
            escaped_group = group_name.replace("\\", "\\\\").replace('"', '\\"')

            if media_path and Path(media_path).exists():
                script = f'''
                    tell application "Messages"
                        set targetChat to chat "{escaped_group}"
                        send POSIX file "{media_path}" to targetChat
                        if "{escaped_text}" is not "" then
                            send "{escaped_text}" to targetChat
                        end if
                    end tell
                '''
            else:
                script = f'''
                    tell application "Messages"
                        set targetChat to chat "{escaped_group}"
                        send "{escaped_text}" to targetChat
                    end tell
                '''

            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=30
            )

            return result.returncode == 0

        except Exception as e:
            logger.error(f"Group send failed: {e}")
            return False

    async def get_recent_chats(self, limit: int = 20) -> List[Dict]:
        """Get recent chat conversations."""
        if not self._db_conn:
            return []

        try:
            cursor = self._db_conn.cursor()
            cursor.execute("""
                SELECT DISTINCT
                    c.chat_identifier,
                    c.display_name,
                    c.group_id,
                    MAX(m.date) as last_message
                FROM chat c
                JOIN chat_message_join cmj ON c.ROWID = cmj.chat_id
                JOIN message m ON cmj.message_id = m.ROWID
                GROUP BY c.ROWID
                ORDER BY last_message DESC
                LIMIT ?
            """, (limit,))

            chats = []
            for row in cursor.fetchall():
                chat_id, display_name, group_id, last_date = row
                chats.append({
                    "chat_id": chat_id,
                    "display_name": display_name or chat_id,
                    "is_group": group_id is not None,
                    "last_message": datetime(2001, 1, 1) + timedelta(seconds=last_date / 1e9) if last_date else None
                })

            return chats

        except Exception as e:
            logger.error(f"Get chats failed: {e}")
            return []

    async def get_chat_history(
        self,
        chat_id: str,
        limit: int = 50
    ) -> List[Dict]:
        """Get message history for a chat."""
        if not self._db_conn:
            return []

        try:
            cursor = self._db_conn.cursor()
            cursor.execute("""
                SELECT
                    m.guid,
                    m.text,
                    m.date,
                    m.is_from_me,
                    h.id as sender
                FROM message m
                LEFT JOIN handle h ON m.handle_id = h.ROWID
                JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                JOIN chat c ON cmj.chat_id = c.ROWID
                WHERE c.chat_identifier = ?
                ORDER BY m.date DESC
                LIMIT ?
            """, (chat_id, limit))

            messages = []
            for row in cursor.fetchall():
                guid, text, date, is_from_me, sender = row
                messages.append({
                    "id": guid,
                    "text": text,
                    "timestamp": datetime(2001, 1, 1) + timedelta(seconds=date / 1e9) if date else None,
                    "is_from_me": bool(is_from_me),
                    "sender": "me" if is_from_me else sender
                })

            return list(reversed(messages))

        except Exception as e:
            logger.error(f"Get history failed: {e}")
            return []

    async def get_contacts(self) -> List[Dict]:
        """Get iMessage contacts from address book."""
        if not IS_MACOS:
            return []

        try:
            # Use AppleScript to get contacts
            script = '''
                tell application "Contacts"
                    set contactList to {}
                    repeat with p in people
                        set pName to name of p
                        set pPhones to {}
                        set pEmails to {}
                        repeat with ph in phones of p
                            set end of pPhones to value of ph
                        end repeat
                        repeat with em in emails of p
                            set end of pEmails to value of em
                        end repeat
                        set end of contactList to {pName, pPhones, pEmails}
                    end repeat
                    return contactList
                end tell
            '''

            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                return []

            # Parse AppleScript output (simplified)
            # Real implementation would need proper parsing
            return []

        except Exception as e:
            logger.error(f"Get contacts failed: {e}")
            return []

    async def start_new_chat(self, participants: List[str]) -> bool:
        """Start a new chat (group or individual)."""
        if not IS_MACOS or not participants:
            return False

        try:
            if len(participants) == 1:
                # Individual chat - just send a message
                return await self.send_message(participants[0], "")
            else:
                # Group chat creation via AppleScript
                buddies = ", ".join([f'buddy "{p}"' for p in participants])
                script = f'''
                    tell application "Messages"
                        set targetService to id of 1st service whose service type = iMessage
                        set newChat to make new text chat with properties {{participants:{{ {buddies} }} of service id targetService}}
                    end tell
                '''

                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                return result.returncode == 0

        except Exception as e:
            logger.error(f"Start chat failed: {e}")
            return False

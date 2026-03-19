"""
Cosmos Slack Channel Adapter
=================================

Slack integration for the Cosmos swarm using Socket Mode.

Features:
- Socket Mode (no public URL needed)
- Slash commands
- App mentions
- Message shortcuts
- Interactive components (buttons, modals)
- File uploads
- Thread replies
- Workflow steps

Based on slack-bolt library.

Setup:
1. Create Slack app at api.slack.com/apps
2. Enable Socket Mode
3. Add Bot Token Scopes: chat:write, app_mentions:read, channels:history, etc.
4. Set SLACK_BOT_TOKEN and SLACK_APP_TOKEN env vars
5. Install app to workspace

"Slack: where work happens. Now with AGI assistance." - The Collective
"""

import os
import asyncio
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
from pathlib import Path
from loguru import logger

from .channel_hub import (
    BaseChannel,
    ChannelConfig,
    ChannelMessage,
    ChannelType,
)

# Slack libraries
try:
    from slack_bolt.async_app import AsyncApp
    from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
    from slack_sdk.web.async_client import AsyncWebClient
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    logger.warning("slack-bolt not installed. Run: pip install slack-bolt aiohttp")


class SlackChannel(BaseChannel):
    """
    Slack channel adapter using Socket Mode.

    Supports:
    - Direct messages
    - Channel messages (when mentioned)
    - Threads
    - Slash commands
    - Interactive components
    - File sharing
    - Reactions
    """

    def __init__(
        self,
        config: ChannelConfig = None,
        bot_token: str = None,
        app_token: str = None
    ):
        """
        Initialize Slack channel.

        Args:
            config: Channel configuration
            bot_token: Bot OAuth token (xoxb-...)
            app_token: App-level token for Socket Mode (xapp-...)
        """
        config = config or ChannelConfig(channel_type=ChannelType.SLACK)
        super().__init__(config)

        self.bot_token = bot_token or os.environ.get("SLACK_BOT_TOKEN")
        self.app_token = app_token or os.environ.get("SLACK_APP_TOKEN")

        self._app: Optional[AsyncApp] = None
        self._handler: Optional[AsyncSocketModeHandler] = None
        self._client: Optional[AsyncWebClient] = None
        self._bot_user_id: Optional[str] = None
        self._slash_handlers: Dict[str, Callable] = {}

    async def connect(self) -> bool:
        """Connect to Slack via Socket Mode."""
        if not SLACK_AVAILABLE:
            logger.error("slack-bolt not available")
            return False

        if not self.bot_token or not self.app_token:
            logger.error("Slack tokens not configured (need both bot and app tokens)")
            return False

        try:
            # Create Bolt app
            self._app = AsyncApp(token=self.bot_token)
            self._client = self._app.client

            # Get bot user ID
            auth = await self._client.auth_test()
            self._bot_user_id = auth["user_id"]

            # Register event handlers
            @self._app.event("message")
            async def handle_message(event, say, client):
                await self._handle_slack_message(event, say, client)

            @self._app.event("app_mention")
            async def handle_mention(event, say, client):
                await self._handle_slack_message(event, say, client, is_mention=True)

            @self._app.command("/cosmos")
            async def handle_cosmos_cmd(ack, command, respond):
                await ack()
                msg = ChannelMessage(
                    message_id=command.get("trigger_id", ""),
                    channel_type=ChannelType.SLACK,
                    channel_id=command["channel_id"],
                    sender_id=command["user_id"],
                    sender_name=command.get("user_name", "Unknown"),
                    text=command.get("text", ""),
                    is_group=True,
                    timestamp=datetime.now(),
                    raw_data={"command": "/cosmos"}
                )
                await self._handle_inbound(msg)

            @self._app.command("/status")
            async def handle_status_cmd(ack, command, respond):
                await ack()
                await respond({
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "*✅ Cosmos Status*\n\n• Swarm: Online\n• Slack: Connected\n• Agents: Active"
                            }
                        }
                    ]
                })

            @self._app.action("cosmos_action")
            async def handle_action(ack, body, client):
                await ack()
                await self._handle_interaction(body, client)

            # Start Socket Mode handler
            self._handler = AsyncSocketModeHandler(self._app, self.app_token)
            asyncio.create_task(self._handler.start_async())

            self._connected = True
            logger.info(f"Slack connected as bot user: {self._bot_user_id}")

            return True

        except Exception as e:
            logger.error(f"Slack connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Slack."""
        if self._handler:
            try:
                await self._handler.close_async()
            except Exception as e:
                logger.warning(f"Slack disconnect error: {e}")

        self._connected = False
        logger.info("Slack disconnected")

    async def send_message(
        self,
        channel_id: str,
        text: str,
        media_path: Optional[str] = None,
        reply_to: Optional[str] = None,
        blocks: List[Dict] = None,
        attachments: List[Dict] = None,
        **kwargs
    ) -> bool:
        """
        Send a message to a Slack channel.

        Args:
            channel_id: Slack channel or user ID
            text: Message text (also used as fallback)
            media_path: Optional file path to upload
            reply_to: Thread timestamp to reply in
            blocks: Block Kit blocks
            attachments: Legacy attachments

        Returns:
            True if sent successfully
        """
        if not self._client:
            return False

        if not self._check_rate_limit(channel_id):
            logger.warning(f"Rate limit exceeded for channel {channel_id}")
            return False

        try:
            # Upload file if provided
            if media_path and Path(media_path).exists():
                result = await self._client.files_upload_v2(
                    channels=channel_id,
                    file=media_path,
                    initial_comment=text,
                    thread_ts=reply_to
                )
                return result.get("ok", False)

            # Send message
            result = await self._client.chat_postMessage(
                channel=channel_id,
                text=text,
                blocks=blocks,
                attachments=attachments,
                thread_ts=reply_to,
                unfurl_links=True,
                unfurl_media=True
            )

            return result.get("ok", False)

        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            return False

    async def _handle_slack_message(
        self,
        event: Dict,
        say: Callable,
        client: AsyncWebClient,
        is_mention: bool = False
    ):
        """Handle incoming Slack message."""
        # Ignore bot messages
        if event.get("bot_id") or event.get("user") == self._bot_user_id:
            return

        # Ignore message subtypes (edits, deletions, etc.) except file_share
        subtype = event.get("subtype")
        if subtype and subtype not in ["file_share", "thread_broadcast"]:
            return

        channel_id = event.get("channel", "")
        user_id = event.get("user", "unknown")
        text = event.get("text", "")

        # Get user info for name
        sender_name = "Unknown"
        try:
            user_info = await client.users_info(user=user_id)
            if user_info.get("ok"):
                profile = user_info["user"].get("profile", {})
                sender_name = profile.get("display_name") or profile.get("real_name") or user_id
        except Exception:
            pass

        # Check if DM or channel
        is_dm = event.get("channel_type") == "im"

        # In channels, only respond to mentions (unless configured otherwise)
        if not is_dm and not is_mention and not self.config.respond_to_all:
            return

        # Build normalized message
        message = ChannelMessage(
            message_id=event.get("ts", ""),
            channel_type=ChannelType.SLACK,
            channel_id=channel_id,
            sender_id=user_id,
            sender_name=sender_name,
            text=text,
            is_group=not is_dm,
            is_mention=is_mention,
            timestamp=datetime.fromtimestamp(float(event.get("ts", 0))),
            raw_data={
                "thread_ts": event.get("thread_ts"),
                "team": event.get("team"),
            }
        )

        # Handle thread replies
        if event.get("thread_ts"):
            message.reply_to_id = event["thread_ts"]

        # Handle files
        files = event.get("files", [])
        if files:
            file = files[0]
            mimetype = file.get("mimetype", "")
            if mimetype.startswith("image"):
                message.media_type = "image"
            elif mimetype.startswith("video"):
                message.media_type = "video"
            elif mimetype.startswith("audio"):
                message.media_type = "audio"
            else:
                message.media_type = "document"
            message.media_url = file.get("url_private")

        # Auto-react if configured
        if self.config.auto_react:
            try:
                await client.reactions_add(
                    channel=channel_id,
                    timestamp=event.get("ts"),
                    name=self.config.auto_react_emoji.replace(":", "")
                )
            except Exception:
                pass

        # Pass to handler
        await self._handle_inbound(message)

    async def _handle_interaction(self, body: Dict, client: AsyncWebClient):
        """Handle interactive component interactions."""
        action = body.get("actions", [{}])[0]
        action_id = action.get("action_id", "")
        user = body.get("user", {})
        channel = body.get("channel", {})

        # Create message from interaction
        message = ChannelMessage(
            message_id=action.get("action_ts", ""),
            channel_type=ChannelType.SLACK,
            channel_id=channel.get("id", ""),
            sender_id=user.get("id", ""),
            sender_name=user.get("name", "Unknown"),
            text=f"[Action: {action_id}] {action.get('value', '')}",
            timestamp=datetime.now(),
            raw_data={"action": action, "body": body}
        )

        await self._handle_inbound(message)

    async def send_blocks(
        self,
        channel_id: str,
        blocks: List[Dict],
        text: str = "Message from Cosmos",
        thread_ts: str = None
    ) -> bool:
        """Send a message with Block Kit blocks."""
        return await self.send_message(
            channel_id,
            text=text,
            blocks=blocks,
            reply_to=thread_ts
        )

    async def send_ephemeral(
        self,
        channel_id: str,
        user_id: str,
        text: str,
        blocks: List[Dict] = None
    ) -> bool:
        """Send an ephemeral message visible only to one user."""
        if not self._client:
            return False

        try:
            result = await self._client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text=text,
                blocks=blocks
            )
            return result.get("ok", False)
        except Exception as e:
            logger.error(f"Ephemeral send failed: {e}")
            return False

    async def update_message(
        self,
        channel_id: str,
        message_ts: str,
        text: str,
        blocks: List[Dict] = None
    ) -> bool:
        """Update an existing message."""
        if not self._client:
            return False

        try:
            result = await self._client.chat_update(
                channel=channel_id,
                ts=message_ts,
                text=text,
                blocks=blocks
            )
            return result.get("ok", False)
        except Exception as e:
            logger.error(f"Message update failed: {e}")
            return False

    async def delete_message(self, channel_id: str, message_ts: str) -> bool:
        """Delete a message."""
        if not self._client:
            return False

        try:
            result = await self._client.chat_delete(
                channel=channel_id,
                ts=message_ts
            )
            return result.get("ok", False)
        except Exception as e:
            logger.error(f"Message delete failed: {e}")
            return False

    async def open_modal(
        self,
        trigger_id: str,
        view: Dict
    ) -> bool:
        """Open a modal dialog."""
        if not self._client:
            return False

        try:
            result = await self._client.views_open(
                trigger_id=trigger_id,
                view=view
            )
            return result.get("ok", False)
        except Exception as e:
            logger.error(f"Modal open failed: {e}")
            return False

    async def get_channel_info(self, channel_id: str) -> Optional[Dict]:
        """Get information about a channel."""
        if not self._client:
            return None

        try:
            result = await self._client.conversations_info(channel=channel_id)
            if result.get("ok"):
                channel = result["channel"]
                return {
                    "id": channel["id"],
                    "name": channel.get("name"),
                    "is_private": channel.get("is_private", False),
                    "is_archived": channel.get("is_archived", False),
                    "topic": channel.get("topic", {}).get("value"),
                    "purpose": channel.get("purpose", {}).get("value"),
                    "member_count": channel.get("num_members", 0)
                }
            return None
        except Exception as e:
            logger.error(f"Get channel info failed: {e}")
            return None

    async def list_channels(self, limit: int = 100) -> List[Dict]:
        """List channels the bot is in."""
        if not self._client:
            return []

        try:
            result = await self._client.conversations_list(
                types="public_channel,private_channel",
                limit=limit
            )
            if result.get("ok"):
                return [
                    {
                        "id": ch["id"],
                        "name": ch.get("name"),
                        "is_private": ch.get("is_private", False)
                    }
                    for ch in result.get("channels", [])
                ]
            return []
        except Exception as e:
            logger.error(f"List channels failed: {e}")
            return []

    def register_slash_command(self, command: str, handler: Callable):
        """Register a custom slash command handler."""
        if self._app:
            self._app.command(command)(handler)
        else:
            self._slash_handlers[command] = handler

    async def set_status(self, text: str, emoji: str = ":robot_face:"):
        """Set the bot's status."""
        if not self._client:
            return

        try:
            await self._client.users_profile_set(
                profile={
                    "status_text": text,
                    "status_emoji": emoji
                }
            )
        except Exception as e:
            logger.error(f"Set status failed: {e}")

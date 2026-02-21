"""
Slack platform adapter.

Uses slack-bolt (Python) with Socket Mode for:
- Receiving messages from channels and DMs
- Sending responses back
- Handling slash commands
- Thread support
"""

import asyncio
import os
from typing import Dict, List, Optional, Any

try:
    from slack_bolt.async_app import AsyncApp
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    from slack_sdk.web.async_client import AsyncWebClient
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    AsyncApp = Any
    AsyncSocketModeHandler = Any
    AsyncWebClient = Any

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_url,
    cache_audio_from_url,
)


def check_slack_requirements() -> bool:
    """Check if Slack dependencies are available."""
    return SLACK_AVAILABLE


class SlackAdapter(BasePlatformAdapter):
    """
    Slack bot adapter using Socket Mode.

    Requires two tokens:
      - SLACK_BOT_TOKEN (xoxb-...) for API calls
      - SLACK_APP_TOKEN (xapp-...) for Socket Mode connection

    Features:
      - DMs and channel messages (mention-gated in channels)
      - Thread support
      - File/image/audio attachments
      - Slash commands (/hermes)
      - Typing indicators (not natively supported by Slack bots)
    """

    MAX_MESSAGE_LENGTH = 4000  # Slack's limit is higher but mrkdwn can inflate

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SLACK)
        self._app: Optional[AsyncApp] = None
        self._handler: Optional[AsyncSocketModeHandler] = None
        self._bot_user_id: Optional[str] = None

    async def connect(self) -> bool:
        """Connect to Slack via Socket Mode."""
        if not SLACK_AVAILABLE:
            print("[Slack] slack-bolt not installed. Run: pip install slack-bolt")
            return False

        bot_token = self.config.token
        app_token = os.getenv("SLACK_APP_TOKEN")

        if not bot_token:
            print("[Slack] SLACK_BOT_TOKEN not set")
            return False
        if not app_token:
            print("[Slack] SLACK_APP_TOKEN not set")
            return False

        try:
            self._app = AsyncApp(token=bot_token)

            # Get our own bot user ID for mention detection
            auth_response = await self._app.client.auth_test()
            self._bot_user_id = auth_response.get("user_id")
            bot_name = auth_response.get("user", "unknown")

            # Register message event handler
            @self._app.event("message")
            async def handle_message_event(event, say):
                await self._handle_slack_message(event)

            # Register slash command handler
            @self._app.command("/hermes")
            async def handle_hermes_command(ack, command):
                await ack()
                await self._handle_slash_command(command)

            # Start Socket Mode handler in background
            self._handler = AsyncSocketModeHandler(self._app, app_token)
            asyncio.create_task(self._handler.start_async())

            self._running = True
            print(f"[Slack] Connected as @{bot_name} (Socket Mode)")
            return True

        except Exception as e:
            print(f"[Slack] Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Slack."""
        if self._handler:
            await self._handler.close_async()
        self._running = False
        print("[Slack] Disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message to a Slack channel or DM."""
        if not self._app:
            return SendResult(success=False, error="Not connected")

        try:
            kwargs = {
                "channel": chat_id,
                "text": content,
            }

            # Reply in thread if thread_ts is available
            if reply_to:
                kwargs["thread_ts"] = reply_to
            elif metadata and metadata.get("thread_ts"):
                kwargs["thread_ts"] = metadata["thread_ts"]

            result = await self._app.client.chat_postMessage(**kwargs)

            return SendResult(
                success=True,
                message_id=result.get("ts"),
                raw_response=result,
            )

        except Exception as e:
            print(f"[Slack] Send error: {e}")
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str) -> None:
        """Slack doesn't have a direct typing indicator API for bots."""
        pass

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """Send an image to Slack by uploading the URL as a file."""
        if not self._app:
            return SendResult(success=False, error="Not connected")

        try:
            import httpx

            # Download the image first
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(image_url)
                response.raise_for_status()

            result = await self._app.client.files_upload_v2(
                channel=chat_id,
                content=response.content,
                filename="image.png",
                initial_comment=caption or "",
                thread_ts=reply_to,
            )

            return SendResult(success=True, raw_response=result)

        except Exception as e:
            # Fall back to sending the URL as text
            text = f"{caption}\n{image_url}" if caption else image_url
            return await self.send(chat_id=chat_id, content=text, reply_to=reply_to)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """Send an audio file to Slack."""
        if not self._app:
            return SendResult(success=False, error="Not connected")

        try:
            result = await self._app.client.files_upload_v2(
                channel=chat_id,
                file=audio_path,
                filename=os.path.basename(audio_path),
                initial_comment=caption or "",
                thread_ts=reply_to,
            )
            return SendResult(success=True, raw_response=result)

        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a Slack channel."""
        if not self._app:
            return {"name": chat_id, "type": "unknown"}

        try:
            result = await self._app.client.conversations_info(channel=chat_id)
            channel = result.get("channel", {})
            is_dm = channel.get("is_im", False)
            return {
                "name": channel.get("name", chat_id),
                "type": "dm" if is_dm else "group",
            }
        except Exception:
            return {"name": chat_id, "type": "unknown"}

    # ----- Internal handlers -----

    async def _handle_slack_message(self, event: dict) -> None:
        """Handle an incoming Slack message event."""
        # Ignore bot messages (including our own)
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        # Ignore message edits and deletions
        subtype = event.get("subtype")
        if subtype in ("message_changed", "message_deleted"):
            return

        text = event.get("text", "")
        user_id = event.get("user", "")
        channel_id = event.get("channel", "")
        thread_ts = event.get("thread_ts") or event.get("ts")
        ts = event.get("ts", "")

        # Determine if this is a DM or channel message
        channel_type = event.get("channel_type", "")
        is_dm = channel_type == "im"

        # In channels, only respond if bot is mentioned
        if not is_dm and self._bot_user_id:
            if f"<@{self._bot_user_id}>" not in text:
                return
            # Strip the bot mention from the text
            text = text.replace(f"<@{self._bot_user_id}>", "").strip()

        # Determine message type
        msg_type = MessageType.TEXT
        if text.startswith("/"):
            msg_type = MessageType.COMMAND

        # Handle file attachments
        media_urls = []
        media_types = []
        files = event.get("files", [])
        for f in files:
            mimetype = f.get("mimetype", "unknown")
            url = f.get("url_private_download") or f.get("url_private", "")
            if mimetype.startswith("image/") and url:
                try:
                    ext = "." + mimetype.split("/")[-1].split(";")[0]
                    if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                        ext = ".jpg"
                    # Slack private URLs require the bot token as auth header
                    cached = await self._download_slack_file(url, ext)
                    media_urls.append(cached)
                    media_types.append(mimetype)
                    msg_type = MessageType.PHOTO
                except Exception as e:
                    print(f"[Slack] Failed to cache image: {e}", flush=True)
            elif mimetype.startswith("audio/") and url:
                try:
                    ext = "." + mimetype.split("/")[-1].split(";")[0]
                    if ext not in (".ogg", ".mp3", ".wav", ".webm", ".m4a"):
                        ext = ".ogg"
                    cached = await self._download_slack_file(url, ext, audio=True)
                    media_urls.append(cached)
                    media_types.append(mimetype)
                    msg_type = MessageType.VOICE
                except Exception as e:
                    print(f"[Slack] Failed to cache audio: {e}", flush=True)

        # Build source
        source = self.build_source(
            chat_id=channel_id,
            chat_name=channel_id,  # Will be resolved later if needed
            chat_type="dm" if is_dm else "group",
            user_id=user_id,
            thread_id=thread_ts,
        )

        msg_event = MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            raw_message=event,
            message_id=ts,
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=thread_ts if thread_ts != ts else None,
        )

        await self.handle_message(msg_event)

    async def _handle_slash_command(self, command: dict) -> None:
        """Handle /hermes slash command."""
        text = command.get("text", "").strip()
        user_id = command.get("user_id", "")
        channel_id = command.get("channel_id", "")

        # Map subcommands to gateway commands
        subcommand_map = {
            "new": "/reset", "reset": "/reset",
            "status": "/status", "stop": "/stop",
            "help": "/help",
            "model": "/model", "personality": "/personality",
            "retry": "/retry", "undo": "/undo",
        }
        first_word = text.split()[0] if text else ""
        if first_word in subcommand_map:
            # Preserve arguments after the subcommand
            rest = text[len(first_word):].strip()
            text = f"{subcommand_map[first_word]} {rest}".strip() if rest else subcommand_map[first_word]
        elif text:
            pass  # Treat as a regular question
        else:
            text = "/help"

        source = self.build_source(
            chat_id=channel_id,
            chat_type="dm",  # Slash commands are always in DM-like context
            user_id=user_id,
        )

        event = MessageEvent(
            text=text,
            message_type=MessageType.COMMAND if text.startswith("/") else MessageType.TEXT,
            source=source,
            raw_message=command,
        )

        await self.handle_message(event)

    async def _download_slack_file(self, url: str, ext: str, audio: bool = False) -> str:
        """Download a Slack file using the bot token for auth."""
        import httpx

        bot_token = self.config.token
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={"Authorization": f"Bearer {bot_token}"},
            )
            response.raise_for_status()

        if audio:
            from gateway.platforms.base import cache_audio_from_bytes
            return cache_audio_from_bytes(response.content, ext)
        else:
            from gateway.platforms.base import cache_image_from_bytes
            return cache_image_from_bytes(response.content, ext)

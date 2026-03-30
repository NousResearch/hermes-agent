"""
Slack platform adapter.

Uses slack-bolt (Python) with Socket Mode for:
- Receiving messages from channels and DMs
- Sending responses back
- Handling slash commands
- Thread support
"""

import asyncio
import logging
import os
import re
from typing import Dict, Optional, Any

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
    SUPPORTED_DOCUMENT_TYPES,
    cache_document_from_bytes,
)


logger = logging.getLogger(__name__)


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

    MAX_MESSAGE_LENGTH = 39000  # Slack API allows 40,000 chars; leave margin

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SLACK)
        self._app: Optional[AsyncApp] = None
        self._handler: Optional[AsyncSocketModeHandler] = None
        self._bot_user_id: Optional[str] = None
        self._user_name_cache: Dict[str, str] = {}  # user_id → display name
        self._socket_mode_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Connect to Slack via Socket Mode."""
        if not SLACK_AVAILABLE:
            logger.error(
                "[Slack] slack-bolt not installed. Run: pip install slack-bolt",
            )
            return False

        bot_token = self.config.token
        app_token = os.getenv("SLACK_APP_TOKEN")

        if not bot_token:
            logger.error("[Slack] SLACK_BOT_TOKEN not set")
            return False
        if not app_token:
            logger.error("[Slack] SLACK_APP_TOKEN not set")
            return False

        try:
            # Acquire scoped lock to prevent duplicate app token usage
            from gateway.status import acquire_scoped_lock
            self._token_lock_identity = app_token
            acquired, existing = acquire_scoped_lock('slack-app-token', app_token, metadata={'platform': 'slack'})
            if not acquired:
                owner_pid = existing.get('pid') if isinstance(existing, dict) else None
                message = f'Slack app token already in use' + (f' (PID {owner_pid})' if owner_pid else '') + '. Stop the other gateway first.'
                logger.error('[%s] %s', self.name, message)
                self._set_fatal_error('slack_token_lock', message, retryable=False)
                return False

            self._app = AsyncApp(token=bot_token)

            # Get our own bot user ID for mention detection
            auth_response = await self._app.client.auth_test()
            self._bot_user_id = auth_response.get("user_id")
            bot_name = auth_response.get("user", "unknown")

            # Register message event handler
            @self._app.event("message")
            async def handle_message_event(event, say):
                await self._handle_slack_message(event)

            # Acknowledge app_mention events to prevent Bolt 404 errors.
            # The "message" handler above already processes @mentions in
            # channels, so this is intentionally a no-op to avoid duplicates.
            @self._app.event("app_mention")
            async def handle_app_mention(event, say):
                pass

            # Register slash command handler
            @self._app.command("/hermes")
            async def handle_hermes_command(ack, command):
                await ack()
                await self._handle_slash_command(command)

            # Start Socket Mode handler in background
            self._handler = AsyncSocketModeHandler(self._app, app_token)
            self._socket_mode_task = asyncio.create_task(self._handler.start_async())

            self._running = True
            logger.info("[Slack] Connected as @%s (Socket Mode)", bot_name)
            return True

        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[Slack] Connection failed: %s", e, exc_info=True)
            return False

    async def disconnect(self) -> None:
        """Disconnect from Slack."""
        if self._handler:
            try:
                await self._handler.close_async()
            except Exception as e:  # pragma: no cover - defensive logging
                logger.warning("[Slack] Error while closing Socket Mode handler: %s", e, exc_info=True)
        self._running = False

        # Release the token lock (use stored identity, not re-read env)
        try:
            from gateway.status import release_scoped_lock
            if getattr(self, '_token_lock_identity', None):
                release_scoped_lock('slack-app-token', self._token_lock_identity)
                self._token_lock_identity = None
        except Exception:
            pass

        logger.info("[Slack] Disconnected")

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
            # Convert standard markdown → Slack mrkdwn
            formatted = self.format_message(content)

            # Split long messages, preserving code block boundaries
            chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)

            thread_ts = self._resolve_thread_ts(reply_to, metadata)
            last_result = None

            # reply_broadcast: also post thread replies to the main channel.
            # Controlled via platform config: gateway.slack.reply_broadcast
            broadcast = self.config.extra.get("reply_broadcast", False)

            for i, chunk in enumerate(chunks):
                kwargs = {
                    "channel": chat_id,
                    "text": chunk,
                }
                if thread_ts:
                    kwargs["thread_ts"] = thread_ts
                    # Only broadcast the first chunk of the first reply
                    if broadcast and i == 0:
                        kwargs["reply_broadcast"] = True

                last_result = await self._app.client.chat_postMessage(**kwargs)

            return SendResult(
                success=True,
                message_id=last_result.get("ts") if last_result else None,
                raw_response=last_result,
            )

        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[Slack] Send error: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
    ) -> SendResult:
        """Edit a previously sent Slack message."""
        if not self._app:
            return SendResult(success=False, error="Not connected")
        try:
            await self._app.client.chat_update(
                channel=chat_id,
                ts=message_id,
                text=content,
            )
            return SendResult(success=True, message_id=message_id)
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(
                "[Slack] Failed to edit message %s in channel %s: %s",
                message_id,
                chat_id,
                e,
                exc_info=True,
            )
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Show a typing/status indicator using assistant.threads.setStatus.

        Displays "is thinking..." next to the bot name in a thread.
        Requires the assistant:write or chat:write scope.
        Auto-clears when the bot sends a reply to the thread.
        """
        if not self._app:
            return

        thread_ts = None
        if metadata:
            thread_ts = metadata.get("thread_id") or metadata.get("thread_ts")

        if not thread_ts:
            return  # Can only set status in a thread context

        try:
            await self._app.client.assistant_threads_setStatus(
                channel_id=chat_id,
                thread_ts=thread_ts,
                status="is thinking...",
            )
        except Exception as e:
            # Silently ignore — may lack assistant:write scope or not be
            # in an assistant-enabled context. Falls back to reactions.
            logger.debug("[Slack] assistant.threads.setStatus failed: %s", e)

    def _resolve_thread_ts(
        self,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Resolve the correct thread_ts for a Slack API call.

        Prefers metadata thread_id (the thread parent's ts, set by the
        gateway) over reply_to (which may be a child message's ts).
        """
        if metadata:
            if metadata.get("thread_id"):
                return metadata["thread_id"]
            if metadata.get("thread_ts"):
                return metadata["thread_ts"]
        return reply_to

    async def _upload_file(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload a local file to Slack."""
        if not self._app:
            return SendResult(success=False, error="Not connected")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        result = await self._app.client.files_upload_v2(
            channel=chat_id,
            file=file_path,
            filename=os.path.basename(file_path),
            initial_comment=caption or "",
            thread_ts=self._resolve_thread_ts(reply_to, metadata),
        )
        return SendResult(success=True, raw_response=result)

    # ----- Markdown → mrkdwn conversion -----

    def format_message(self, content: str) -> str:
        """Convert standard markdown to Slack mrkdwn format.

        Protected regions (code blocks, inline code) are extracted first so
        their contents are never modified.  Standard markdown constructs
        (headers, bold, italic, links) are translated to mrkdwn syntax.
        """
        if not content:
            return content

        placeholders: dict = {}
        counter = [0]

        def _ph(value: str) -> str:
            """Stash value behind a placeholder that survives later passes."""
            key = f"\x00SL{counter[0]}\x00"
            counter[0] += 1
            placeholders[key] = value
            return key

        text = content

        # 1) Protect fenced code blocks (``` ... ```)
        text = re.sub(
            r'(```(?:[^\n]*\n)?[\s\S]*?```)',
            lambda m: _ph(m.group(0)),
            text,
        )

        # 2) Protect inline code (`...`)
        text = re.sub(r'(`[^`]+`)', lambda m: _ph(m.group(0)), text)

        # 3) Convert markdown links [text](url) → <url|text>
        text = re.sub(
            r'\[([^\]]+)\]\(([^)]+)\)',
            lambda m: _ph(f'<{m.group(2)}|{m.group(1)}>'),
            text,
        )

        # 4) Convert headers (## Title) → *Title* (bold)
        def _convert_header(m):
            inner = m.group(1).strip()
            # Strip redundant bold markers inside a header
            inner = re.sub(r'\*\*(.+?)\*\*', r'\1', inner)
            return _ph(f'*{inner}*')

        text = re.sub(
            r'^#{1,6}\s+(.+)$', _convert_header, text, flags=re.MULTILINE
        )

        # 5) Convert bold: **text** → *text* (Slack bold)
        text = re.sub(
            r'\*\*(.+?)\*\*',
            lambda m: _ph(f'*{m.group(1)}*'),
            text,
        )

        # 6) Convert italic: _text_ stays as _text_ (already Slack italic)
        #    Single *text* → _text_ (Slack italic)
        text = re.sub(
            r'(?<!\*)\*([^*\n]+)\*(?!\*)',
            lambda m: _ph(f'_{m.group(1)}_'),
            text,
        )

        # 7) Convert strikethrough: ~~text~~ → ~text~
        text = re.sub(
            r'~~(.+?)~~',
            lambda m: _ph(f'~{m.group(1)}~'),
            text,
        )

        # 8) Convert blockquotes: > text → > text (same syntax, just ensure
        #    no extra escaping happens to the > character)
        # Slack uses the same > prefix, so this is a no-op for content.

        # 9) Restore placeholders in reverse order
        for key in reversed(list(placeholders.keys())):
            text = text.replace(key, placeholders[key])

        return text

    # ----- Reactions -----

    async def _add_reaction(
        self, channel: str, timestamp: str, emoji: str
    ) -> bool:
        """Add an emoji reaction to a message. Returns True on success."""
        if not self._app:
            return False
        try:
            await self._app.client.reactions_add(
                channel=channel, timestamp=timestamp, name=emoji
            )
            return True
        except Exception as e:
            # Don't log as error — may fail if already reacted or missing scope
            logger.debug("[Slack] reactions.add failed (%s): %s", emoji, e)
            return False

    async def _remove_reaction(
        self, channel: str, timestamp: str, emoji: str
    ) -> bool:
        """Remove an emoji reaction from a message. Returns True on success."""
        if not self._app:
            return False
        try:
            await self._app.client.reactions_remove(
                channel=channel, timestamp=timestamp, name=emoji
            )
            return True
        except Exception as e:
            logger.debug("[Slack] reactions.remove failed (%s): %s", emoji, e)
            return False

    # ----- User identity resolution -----

    async def _resolve_user_name(self, user_id: str) -> str:
        """Resolve a Slack user ID to a display name, with caching."""
        if not user_id:
            return ""
        if user_id in self._user_name_cache:
            return self._user_name_cache[user_id]

        if not self._app:
            return user_id

        try:
            result = await self._app.client.users_info(user=user_id)
            user = result.get("user", {})
            # Prefer display_name → real_name → user_id
            profile = user.get("profile", {})
            name = (
                profile.get("display_name")
                or profile.get("real_name")
                or user.get("real_name")
                or user.get("name")
                or user_id
            )
            self._user_name_cache[user_id] = name
            return name
        except Exception as e:
            logger.debug("[Slack] users.info failed for %s: %s", user_id, e)
            self._user_name_cache[user_id] = user_id
            return user_id

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a local image file to Slack by uploading it."""
        try:
            return await self._upload_file(chat_id, image_path, caption, reply_to, metadata)
        except FileNotFoundError:
            return SendResult(success=False, error=f"Image file not found: {image_path}")
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(
                "[%s] Failed to send local Slack image %s: %s",
                self.name,
                image_path,
                e,
                exc_info=True,
            )
            text = f"🖼️ Image: {image_path}"
            if caption:
                text = f"{caption}\n{text}"
            return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
                thread_ts=self._resolve_thread_ts(reply_to, metadata),
            )

            return SendResult(success=True, raw_response=result)

        except Exception as e:  # pragma: no cover - defensive logging
            logger.warning(
                "[Slack] Failed to upload image from URL %s, falling back to text: %s",
                image_url,
                e,
                exc_info=True,
            )
            # Fall back to sending the URL as text
            text = f"{caption}\n{image_url}" if caption else image_url
            return await self.send(chat_id=chat_id, content=text, reply_to=reply_to)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send an audio file to Slack."""
        try:
            return await self._upload_file(chat_id, audio_path, caption, reply_to, metadata)
        except FileNotFoundError:
            return SendResult(success=False, error=f"Audio file not found: {audio_path}")
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(
                "[Slack] Failed to send audio file %s: %s",
                audio_path,
                e,
                exc_info=True,
            )
            return SendResult(success=False, error=str(e))

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a video file to Slack."""
        if not self._app:
            return SendResult(success=False, error="Not connected")

        if not os.path.exists(video_path):
            return SendResult(success=False, error=f"Video file not found: {video_path}")

        try:
            result = await self._app.client.files_upload_v2(
                channel=chat_id,
                file=video_path,
                filename=os.path.basename(video_path),
                initial_comment=caption or "",
                thread_ts=self._resolve_thread_ts(reply_to, metadata),
            )
            return SendResult(success=True, raw_response=result)

        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(
                "[%s] Failed to send video %s: %s",
                self.name,
                video_path,
                e,
                exc_info=True,
            )
            text = f"🎬 Video: {video_path}"
            if caption:
                text = f"{caption}\n{text}"
            return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a document/file attachment to Slack."""
        if not self._app:
            return SendResult(success=False, error="Not connected")

        if not os.path.exists(file_path):
            return SendResult(success=False, error=f"File not found: {file_path}")

        display_name = file_name or os.path.basename(file_path)

        try:
            result = await self._app.client.files_upload_v2(
                channel=chat_id,
                file=file_path,
                filename=display_name,
                initial_comment=caption or "",
                thread_ts=self._resolve_thread_ts(reply_to, metadata),
            )
            return SendResult(success=True, raw_response=result)

        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(
                "[%s] Failed to send document %s: %s",
                self.name,
                file_path,
                e,
                exc_info=True,
            )
            text = f"📎 File: {file_path}"
            if caption:
                text = f"{caption}\n{text}"
            return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

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
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(
                "[Slack] Failed to fetch chat info for %s: %s",
                chat_id,
                e,
                exc_info=True,
            )
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
        ts = event.get("ts", "")

        # Determine if this is a DM or channel message
        channel_type = event.get("channel_type", "")
        is_dm = channel_type == "im"

        # Build thread_ts for session keying.
        # In channels: fall back to ts so each top-level @mention starts a
        #   new thread/session (the bot always replies in a thread).
        # In DMs: only use the real thread_ts — top-level DMs should share
        #   one continuous session, threaded DMs get their own session.
        if is_dm:
            thread_ts = event.get("thread_ts")  # None for top-level DMs
        else:
            thread_ts = event.get("thread_ts") or ts  # ts fallback for channels

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
                except Exception as e:  # pragma: no cover - defensive logging
                    logger.warning("[Slack] Failed to cache image from %s: %s", url, e, exc_info=True)
            elif mimetype.startswith("audio/") and url:
                try:
                    ext = "." + mimetype.split("/")[-1].split(";")[0]
                    if ext not in (".ogg", ".mp3", ".wav", ".webm", ".m4a"):
                        ext = ".ogg"
                    cached = await self._download_slack_file(url, ext, audio=True)
                    media_urls.append(cached)
                    media_types.append(mimetype)
                    msg_type = MessageType.VOICE
                except Exception as e:  # pragma: no cover - defensive logging
                    logger.warning("[Slack] Failed to cache audio from %s: %s", url, e, exc_info=True)
            elif url:
                # Try to handle as a document attachment
                try:
                    original_filename = f.get("name", "")
                    ext = ""
                    if original_filename:
                        _, ext = os.path.splitext(original_filename)
                        ext = ext.lower()

                    # Fallback: reverse-lookup from MIME type
                    if not ext and mimetype:
                        mime_to_ext = {v: k for k, v in SUPPORTED_DOCUMENT_TYPES.items()}
                        ext = mime_to_ext.get(mimetype, "")

                    if ext not in SUPPORTED_DOCUMENT_TYPES:
                        continue  # Skip unsupported file types silently

                    # Check file size (Slack limit: 20 MB for bots)
                    file_size = f.get("size", 0)
                    MAX_DOC_BYTES = 20 * 1024 * 1024
                    if not file_size or file_size > MAX_DOC_BYTES:
                        logger.warning("[Slack] Document too large or unknown size: %s", file_size)
                        continue

                    # Download and cache
                    raw_bytes = await self._download_slack_file_bytes(url)
                    cached_path = cache_document_from_bytes(
                        raw_bytes, original_filename or f"document{ext}"
                    )
                    doc_mime = SUPPORTED_DOCUMENT_TYPES[ext]
                    media_urls.append(cached_path)
                    media_types.append(doc_mime)
                    msg_type = MessageType.DOCUMENT
                    logger.debug("[Slack] Cached user document: %s", cached_path)

                    # Inject text content for .txt/.md files (capped at 100 KB)
                    MAX_TEXT_INJECT_BYTES = 100 * 1024
                    if ext in (".md", ".txt") and len(raw_bytes) <= MAX_TEXT_INJECT_BYTES:
                        try:
                            text_content = raw_bytes.decode("utf-8")
                            display_name = original_filename or f"document{ext}"
                            display_name = re.sub(r'[^\w.\- ]', '_', display_name)
                            injection = f"[Content of {display_name}]:\n{text_content}"
                            if text:
                                text = f"{injection}\n\n{text}"
                            else:
                                text = injection
                        except UnicodeDecodeError:
                            pass  # Binary content, skip injection

                except Exception as e:  # pragma: no cover - defensive logging
                    logger.warning("[Slack] Failed to cache document from %s: %s", url, e, exc_info=True)

        # Resolve user display name (cached after first lookup)
        user_name = await self._resolve_user_name(user_id)

        # Build source
        source = self.build_source(
            chat_id=channel_id,
            chat_name=channel_id,  # Will be resolved later if needed
            chat_type="dm" if is_dm else "group",
            user_id=user_id,
            user_name=user_name,
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

        # Add 👀 reaction to acknowledge receipt
        await self._add_reaction(channel_id, ts, "eyes")

        await self.handle_message(msg_event)

        # Replace 👀 with ✅ when done
        await self._remove_reaction(channel_id, ts, "eyes")
        await self._add_reaction(channel_id, ts, "white_check_mark")

    async def _handle_slash_command(self, command: dict) -> None:
        """Handle /hermes slash command."""
        text = command.get("text", "").strip()
        user_id = command.get("user_id", "")
        channel_id = command.get("channel_id", "")

        # Map subcommands to gateway commands — derived from central registry.
        # Also keep "compact" as a Slack-specific alias for /compress.
        from hermes_cli.commands import slack_subcommand_map
        subcommand_map = slack_subcommand_map()
        subcommand_map["compact"] = "/compress"
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
        """Download a Slack file using the bot token for auth, with retry."""
        import asyncio
        import httpx

        bot_token = self.config.token
        last_exc = None

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            for attempt in range(3):
                try:
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
                except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                    last_exc = exc
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 429:
                        raise
                    if attempt < 2:
                        logger.debug("Slack file download retry %d/2 for %s: %s",
                                     attempt + 1, url[:80], exc)
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    raise
        raise last_exc

    async def _download_slack_file_bytes(self, url: str) -> bytes:
        """Download a Slack file and return raw bytes, with retry."""
        import asyncio
        import httpx

        bot_token = self.config.token
        last_exc = None

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            for attempt in range(3):
                try:
                    response = await client.get(
                        url,
                        headers={"Authorization": f"Bearer {bot_token}"},
                    )
                    response.raise_for_status()
                    return response.content
                except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                    last_exc = exc
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 429:
                        raise
                    if attempt < 2:
                        logger.debug("Slack file download retry %d/2 for %s: %s",
                                     attempt + 1, url[:80], exc)
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    raise
        raise last_exc

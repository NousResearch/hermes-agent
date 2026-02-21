"""
WhatsApp platform adapter.

WhatsApp integration is more complex than Telegram/Discord because:
- No official bot API for personal accounts
- Business API requires Meta Business verification
- Most solutions use web-based automation

This adapter supports multiple backends:
1. WhatsApp Business API (requires Meta verification)
2. whatsapp-web.js (via Node.js subprocess) - for personal accounts
3. Baileys (via Node.js subprocess) - alternative for personal accounts

For simplicity, we'll implement a generic interface that can work
with different backends via a bridge pattern.
"""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_url,
    cache_audio_from_url,
)


def check_whatsapp_requirements() -> bool:
    """
    Check if WhatsApp dependencies are available.
    
    WhatsApp requires a Node.js bridge for most implementations.
    """
    # Check for Node.js
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


class WhatsAppAdapter(BasePlatformAdapter):
    """
    WhatsApp adapter.
    
    This implementation uses a simple HTTP bridge pattern where:
    1. A Node.js process runs the WhatsApp Web client
    2. Messages are forwarded via HTTP/IPC to this Python adapter
    3. Responses are sent back through the bridge
    
    The actual Node.js bridge implementation can vary:
    - whatsapp-web.js based
    - Baileys based
    - Business API based
    
    Configuration:
    - bridge_script: Path to the Node.js bridge script
    - bridge_port: Port for HTTP communication (default: 3000)
    - session_path: Path to store WhatsApp session data
    """
    
    # WhatsApp message limits
    MAX_MESSAGE_LENGTH = 65536  # WhatsApp allows longer messages
    
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WHATSAPP)
        self._bridge_process: Optional[subprocess.Popen] = None
        self._bridge_port: int = config.extra.get("bridge_port", 3000)
        self._bridge_script: Optional[str] = config.extra.get("bridge_script")
        self._session_path: Path = Path(config.extra.get(
            "session_path",
            Path.home() / ".hermes" / "whatsapp" / "session"
        ))
        self._message_queue: asyncio.Queue = asyncio.Queue()
    
    async def connect(self) -> bool:
        """
        Start the WhatsApp bridge.
        
        This launches the Node.js bridge process and waits for it to be ready.
        """
        if not check_whatsapp_requirements():
            print(f"[{self.name}] Node.js not found. WhatsApp requires Node.js.")
            return False
        
        if not self._bridge_script:
            print(f"[{self.name}] No bridge script configured.")
            print(f"[{self.name}] Set 'bridge_script' in whatsapp.extra config.")
            print(f"[{self.name}] See docs/messaging.md for WhatsApp setup instructions.")
            return False
        
        bridge_path = Path(self._bridge_script)
        if not bridge_path.exists():
            print(f"[{self.name}] Bridge script not found: {bridge_path}")
            return False
        
        try:
            # Ensure session directory exists
            self._session_path.mkdir(parents=True, exist_ok=True)
            
            # Start the bridge process
            self._bridge_process = subprocess.Popen(
                [
                    "node",
                    str(bridge_path),
                    "--port", str(self._bridge_port),
                    "--session", str(self._session_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            # Wait for bridge to be ready (look for ready signal)
            # This is a simplified version - real implementation would
            # wait for an HTTP health check or specific stdout message
            await asyncio.sleep(5)
            
            if self._bridge_process.poll() is not None:
                stderr = self._bridge_process.stderr.read() if self._bridge_process.stderr else ""
                print(f"[{self.name}] Bridge process died: {stderr}")
                return False
            
            # Start message polling task
            asyncio.create_task(self._poll_messages())
            
            self._running = True
            print(f"[{self.name}] Bridge started on port {self._bridge_port}")
            print(f"[{self.name}] Scan QR code if prompted (check bridge output)")
            return True
            
        except Exception as e:
            print(f"[{self.name}] Failed to start bridge: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Stop the WhatsApp bridge."""
        if self._bridge_process:
            try:
                self._bridge_process.terminate()
                await asyncio.sleep(1)
                if self._bridge_process.poll() is None:
                    self._bridge_process.kill()
            except Exception as e:
                print(f"[{self.name}] Error stopping bridge: {e}")
        
        self._running = False
        self._bridge_process = None
        print(f"[{self.name}] Disconnected")
    
    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        """Send a message via the WhatsApp bridge."""
        if not self._running:
            return SendResult(success=False, error="Not connected")
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "chatId": chat_id,
                    "message": content,
                }
                if reply_to:
                    payload["replyTo"] = reply_to
                
                async with session.post(
                    f"http://localhost:{self._bridge_port}/send",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return SendResult(
                            success=True,
                            message_id=data.get("messageId"),
                            raw_response=data
                        )
                    else:
                        error = await resp.text()
                        return SendResult(success=False, error=error)
                        
        except ImportError:
            return SendResult(
                success=False, 
                error="aiohttp not installed. Run: pip install aiohttp"
            )
        except Exception as e:
            return SendResult(success=False, error=str(e))
    
    async def send_typing(self, chat_id: str) -> None:
        """Send typing indicator via bridge."""
        if not self._running:
            return
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"http://localhost:{self._bridge_port}/typing",
                    json={"chatId": chat_id},
                    timeout=aiohttp.ClientTimeout(total=5)
                )
        except Exception:
            pass  # Ignore typing indicator failures
    
    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a WhatsApp chat."""
        if not self._running:
            return {"name": "Unknown", "type": "dm"}
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{self._bridge_port}/chat/{chat_id}",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "name": data.get("name", chat_id),
                            "type": "group" if data.get("isGroup") else "dm",
                            "participants": data.get("participants", []),
                        }
        except Exception as e:
            logger.debug("Could not get WhatsApp chat info for %s: %s", chat_id, e)
        
        return {"name": chat_id, "type": "dm"}
    
    async def _poll_messages(self) -> None:
        """Poll the bridge for incoming messages."""
        try:
            import aiohttp
        except ImportError:
            print(f"[{self.name}] aiohttp not installed, message polling disabled")
            return
        
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://localhost:{self._bridge_port}/messages",
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status == 200:
                            messages = await resp.json()
                            for msg_data in messages:
                                event = await self._build_message_event(msg_data)
                                if event:
                                    await self.handle_message(event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[{self.name}] Poll error: {e}")
                await asyncio.sleep(5)
            
            await asyncio.sleep(1)  # Poll interval
    
    async def _build_message_event(self, data: Dict[str, Any]) -> Optional[MessageEvent]:
        """Build a MessageEvent from bridge message data, downloading images to cache."""
        try:
            # Determine message type
            msg_type = MessageType.TEXT
            if data.get("hasMedia"):
                media_type = data.get("mediaType", "")
                if "image" in media_type:
                    msg_type = MessageType.PHOTO
                elif "video" in media_type:
                    msg_type = MessageType.VIDEO
                elif "audio" in media_type or "ptt" in media_type:  # ptt = voice note
                    msg_type = MessageType.VOICE
                else:
                    msg_type = MessageType.DOCUMENT
            
            # Determine chat type
            is_group = data.get("isGroup", False)
            chat_type = "group" if is_group else "dm"
            
            # Build source
            source = self.build_source(
                chat_id=data.get("chatId", ""),
                chat_name=data.get("chatName"),
                chat_type=chat_type,
                user_id=data.get("senderId"),
                user_name=data.get("senderName"),
            )
            
            # Download image media URLs to the local cache so the vision tool
            # can access them reliably regardless of URL expiration.
            raw_urls = data.get("mediaUrls", [])
            cached_urls = []
            media_types = []
            for url in raw_urls:
                if msg_type == MessageType.PHOTO and url.startswith(("http://", "https://")):
                    try:
                        cached_path = await cache_image_from_url(url, ext=".jpg")
                        cached_urls.append(cached_path)
                        media_types.append("image/jpeg")
                        print(f"[{self.name}] Cached user image: {cached_path}", flush=True)
                    except Exception as e:
                        print(f"[{self.name}] Failed to cache image: {e}", flush=True)
                        cached_urls.append(url)
                        media_types.append("image/jpeg")
                elif msg_type == MessageType.VOICE and url.startswith(("http://", "https://")):
                    try:
                        cached_path = await cache_audio_from_url(url, ext=".ogg")
                        cached_urls.append(cached_path)
                        media_types.append("audio/ogg")
                        print(f"[{self.name}] Cached user voice: {cached_path}", flush=True)
                    except Exception as e:
                        print(f"[{self.name}] Failed to cache voice: {e}", flush=True)
                        cached_urls.append(url)
                        media_types.append("audio/ogg")
                else:
                    cached_urls.append(url)
                    media_types.append("unknown")
            
            return MessageEvent(
                text=data.get("body", ""),
                message_type=msg_type,
                source=source,
                raw_message=data,
                message_id=data.get("messageId"),
                media_urls=cached_urls,
                media_types=media_types,
            )
        except Exception as e:
            print(f"[{self.name}] Error building event: {e}")
            return None


# Note: A reference Node.js bridge script would be provided in scripts/whatsapp-bridge/
# It would use whatsapp-web.js or Baileys to:
# 1. Handle WhatsApp Web authentication (QR code)
# 2. Listen for incoming messages
# 3. Expose HTTP endpoints for send/receive/status

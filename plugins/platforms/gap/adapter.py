import asyncio
import logging
import os
from typing import Any, Dict, Optional
import aiohttp

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
import hermes_cli.gateway as gateway_mod

logger = logging.getLogger(__name__)

def check_gap_requirements() -> bool:
    try:
        import aiohttp
    except ImportError:
        return False
    return bool(os.getenv("GAP_BOT_TOKEN"))

def _is_connected(config) -> bool:
    return bool((gateway_mod.get_env_value("GAP_BOT_TOKEN") or "").strip())

class GapAdapter(BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = 1000

    def __init__(self, config: PlatformConfig):
        super().__init__(config, "gap")
        self.env_vars = {k: os.getenv(k, "") for k in ['GAP_BOT_TOKEN']}
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self.env_vars["GAP_BOT_TOKEN"]:
            return False
        self._http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        self._running = True
        return True

    async def disconnect(self) -> None:
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
        self._running = False

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        if not self._http_session:
            return SendResult(success=False, error="Not connected")
        try:

        payload = {"chat_id": chat_id, "data": content}
        url = "https://api.gap.im/sendMessage"
        headers = {"token": self.env_vars['GAP_BOT_TOKEN']}
        async with self._http_session.post(url, data=payload, headers=headers) as resp:
            data = await resp.json()
            if "id" not in data: return SendResult(success=False, error=str(data))
            return SendResult(success=True, message_id=str(data.get("id", "")))

        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}

def _build_adapter(config):
    return GapAdapter(config)

def register(ctx) -> None:
    ctx.register_platform(
        name="gap",
        label="Gap (گپ)",
        adapter_factory=_build_adapter,
        check_fn=check_gap_requirements,
        is_connected=_is_connected,
        required_env=["GAP_BOT_TOKEN"],
        install_hint="pip install aiohttp",
        allowed_users_env="GAP_ALLOWED_USERS",
        allow_all_env="GAP_ALLOW_ALL_USERS",
        max_message_length=1000,
        emoji="💬",
        allow_update_command=True,
    )

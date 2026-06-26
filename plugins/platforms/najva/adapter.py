import asyncio
import logging
import os
from typing import Any, Dict, Optional
import aiohttp

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
import hermes_cli.gateway as gateway_mod

logger = logging.getLogger(__name__)

def check_najva_requirements() -> bool:
    try:
        import aiohttp
    except ImportError:
        return False
    return bool(os.getenv("NAJVA_API_TOKEN"))

def _is_connected(config) -> bool:
    return bool((gateway_mod.get_env_value("NAJVA_API_TOKEN") or "").strip())

class NajvaAdapter(BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = 1000

    def __init__(self, config: PlatformConfig):
        super().__init__(config, "najva")
        self.env_vars = {k: os.getenv(k, "") for k in ['NAJVA_API_TOKEN']}
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self.env_vars["NAJVA_API_TOKEN"]:
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

        payload = {"title": "Hermes", "body": content, "subscriber_tokens": [chat_id]}
        url = "https://app.najva.com/api/v1/notifications/"
        headers = {"Authorization": f"Token {self.env_vars['NAJVA_API_TOKEN']}"}
        async with self._http_session.post(url, json=payload, headers=headers) as resp:
            if resp.status >= 400: return SendResult(success=False, error=await resp.text())
            data = await resp.json()
            return SendResult(success=True, message_id=str(data.get("id", "")))

        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}

def _build_adapter(config):
    return NajvaAdapter(config)

def register(ctx) -> None:
    ctx.register_platform(
        name="najva",
        label="Najva (نجوا)",
        adapter_factory=_build_adapter,
        check_fn=check_najva_requirements,
        is_connected=_is_connected,
        required_env=["NAJVA_API_TOKEN"],
        install_hint="pip install aiohttp",
        allowed_users_env="NAJVA_ALLOWED_USERS",
        allow_all_env="NAJVA_ALLOW_ALL_USERS",
        max_message_length=1000,
        emoji="🔔",
        allow_update_command=True,
    )

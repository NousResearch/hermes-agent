import asyncio
import logging
import os
from typing import Any, Dict, Optional
import aiohttp

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
import hermes_cli.gateway as gateway_mod

logger = logging.getLogger(__name__)

def check_kavenegar_requirements() -> bool:
    try:
        import aiohttp
    except ImportError:
        return False
    return bool(os.getenv("KAVENEGAR_API_KEY"))

def _is_connected(config) -> bool:
    return bool((gateway_mod.get_env_value("KAVENEGAR_API_KEY") or "").strip())

class KavenegarAdapter(BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = 1000

    def __init__(self, config: PlatformConfig):
        super().__init__(config, "kavenegar")
        self.env_vars = {k: os.getenv(k, "") for k in ['KAVENEGAR_API_KEY', 'KAVENEGAR_SENDER']}
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self.env_vars["KAVENEGAR_API_KEY"]:
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

        payload = {"receptor": chat_id, "message": content}
        if self.env_vars.get("KAVENEGAR_SENDER"): payload["sender"] = self.env_vars["KAVENEGAR_SENDER"]
        url = f"https://api.kavenegar.com/v1/{self.env_vars['KAVENEGAR_API_KEY']}/sms/send.json"
        async with self._http_session.post(url, data=payload) as resp:
            data = await resp.json()
            if data.get("return", {}).get("status") != 200: return SendResult(success=False, error=str(data))
            entries = data.get("entries", [])
            return SendResult(success=True, message_id=str(entries[0].get("messageid", "")) if entries else "")

        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}

def _build_adapter(config):
    return KavenegarAdapter(config)

def register(ctx) -> None:
    ctx.register_platform(
        name="kavenegar",
        label="Kavenegar (کاوه‌نگار)",
        adapter_factory=_build_adapter,
        check_fn=check_kavenegar_requirements,
        is_connected=_is_connected,
        required_env=["KAVENEGAR_API_KEY"],
        install_hint="pip install aiohttp",
        allowed_users_env="KAVENEGAR_ALLOWED_USERS",
        allow_all_env="KAVENEGAR_ALLOW_ALL_USERS",
        max_message_length=1000,
        emoji="💬",
        allow_update_command=True,
    )

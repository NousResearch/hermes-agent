"""One profile's independently locked iLink accounts and transport configuration watcher."""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from pathlib import Path

from hermes_constants import get_hermes_home
from gateway.config import HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.weixin import WeixinAdapter, check_weixin_requirements, list_weixin_accounts
from gateway.platforms.helpers import cancel_task
from utils import file_signature

logger = logging.getLogger(__name__)

# Installed by the runner before connect, copied whenever an account is rebuilt.
_WIRING = (
    "gateway_runner", "_message_handler", "_fatal_error_handler", "_session_store",
    "_busy_session_handler", "_reaction_handler", "_topic_recovery_fn", "_authorization_check",
    "_platform_event_handler", "_busy_text_mode", "_busy_text_debounce_seconds",
    "_busy_text_hard_cap_seconds", "_human_delay_range_ms", "_owner_profile", "_hermes_profile_name",
)


def account_configs(config: PlatformConfig, home: Path) -> dict[str, PlatformConfig]:
    """YAML holds account references/settings; saved login files hold tokens."""
    if not config.enabled:
        return {}
    saved = {row["account_id"]: row for row in list_weixin_accounts(str(home))}
    common = copy.deepcopy(config.extra or {})
    legacy_token = common.pop("token", None)
    primary = str(common.pop("account_id", "") or "")
    rows = common.pop("accounts", {}) or {}
    common.pop("default_account", None)
    if not isinstance(rows, dict):
        raise ValueError("Weixin extra.accounts must be an account-id mapping")
    rows = copy.deepcopy(rows)
    if primary and primary not in rows:
        rows[primary] = {}
    result = {}
    for account_id, overrides in rows.items():
        if not isinstance(overrides, dict):
            raise ValueError("Weixin account settings must be a mapping")
        if overrides.get("enabled", True) is False:
            continue
        credentials = saved.get(str(account_id), {})
        token = (config.token or legacy_token or credentials.get("token")) if account_id == primary else credentials.get("token")
        if not token:
            logger.warning("Weixin account %s has no saved login in this profile", account_id)
            continue
        extra = {**common, "base_url": common.get("base_url") or credentials.get("base_url"),
                 **overrides, "account_id": str(account_id), "token": token}
        child = copy.deepcopy(config)
        child.token, child.extra = token, extra
        if "home_channel" in overrides:
            child.home_channel = HomeChannel.from_dict(overrides["home_channel"]) if overrides["home_channel"] else None
        result[str(account_id)] = child
    return result


def has_weixin_credentials(config: PlatformConfig) -> bool:
    return bool(account_configs(config, Path(get_hermes_home())))


def resolve_outbound_account(extra, token, chat_id, account_id=None):
    from gateway.session_context import get_session_env
    if "/" in chat_id:
        account_id, chat_id = chat_id.split("/", 1)
    if not account_id and get_session_env("HERMES_SESSION_PLATFORM") == "weixin":
        account_id = get_session_env("HERMES_SESSION_ACCOUNT_ID") or None
    account_id = account_id or extra.get("default_account") or extra.get("account_id")
    configs = account_configs(PlatformConfig(enabled=True, token=token, extra=extra), Path(get_hermes_home()))
    selected = configs.get(account_id)
    if selected is None:
        raise ValueError("Requested Weixin account is disabled or has no login in this profile")
    return selected, chat_id


class WeixinAccountGroup(BasePlatformAdapter):
    SUPPORTS_MESSAGE_EDITING = False
    SUPPORTS_BLOCK_STREAMING = True
    supports_code_blocks = True
    splits_long_messages = True
    MAX_MESSAGE_LENGTH = WeixinAdapter.MAX_MESSAGE_LENGTH

    def __init__(self, config):
        super().__init__(config, Platform.WEIXIN)
        self._home = Path(get_hermes_home())
        self.accounts: dict[str, WeixinAdapter] = {}
        self._account_configs = {}
        self._watch_task = None
        self._signature = None
        self._pending_signature = None
        self._default_account = ""
        self._retry_pending = False
        self._retry_after = 0.0
        self._outbound_inflight = {}

    def owns_transport(self, adapter):
        return adapter is self or any(adapter is child for child in self.accounts.values())

    def resolve_source_adapter(self, source):
        account_id = getattr(source, "account_id", None)
        return self.accounts.get(account_id or self._default_account)

    def _wire_child(self, child):
        for name in _WIRING:
            if hasattr(self, name):
                setattr(child, name, getattr(self, name))
        runner = self.gateway_runner
        if runner is not None:
            child.set_authorization_check(runner._make_adapter_auth_check(
                Platform.WEIXIN, profile_name=self._owner_profile, transport_adapter=child))
            runner._sync_voice_mode_state_to_adapter(child)

    async def connect(self, *, is_reconnect=False):
        await self.apply_config(self.config, is_reconnect=is_reconnect)
        if not self.accounts:
            return False
        self._mark_connected()
        self._signature = self._config_signature()
        self._watch_task = asyncio.create_task(self._watch_config(), name="weixin-config-watch")
        return True

    async def disconnect(self):
        await cancel_task(self._watch_task)
        self._watch_task = None
        for child in list(self.accounts.values()):
            await child.disconnect()
        self.accounts.clear()
        self._account_configs.clear()
        self._mark_disconnected()

    async def apply_config(self, config, *, is_reconnect=False):
        desired = account_configs(config, self._home)
        pending = False
        for account_id in dict.fromkeys([*self.accounts, *desired]):
            child = self.accounts.get(account_id)
            wanted = desired.get(account_id)
            if child is not None and wanted == self._account_configs.get(account_id):
                continue
            if child is not None and self._account_busy(account_id, child):
                pending = True
                continue
            if child is not None:
                await child.disconnect()
                del self.accounts[account_id]
                self._account_configs.pop(account_id, None)
            if wanted is not None:
                replacement = WeixinAdapter(wanted)
                self._wire_child(replacement)
                if child is not None:
                    replacement._dedup = child._dedup
                # Register before polling can dispatch its first message.
                self.accounts[account_id] = replacement
                try:
                    connected = await replacement.connect(is_reconnect=is_reconnect)
                except BaseException:
                    await replacement.disconnect()
                    self.accounts.pop(account_id, None)
                    raise
                if connected:
                    self._account_configs[account_id] = copy.deepcopy(wanted)
                else:
                    await replacement.disconnect()
                    self.accounts.pop(account_id, None)
                    pending = True
        self.config = config
        self._publish_transport_config(config)
        self._default_account = str(config.extra.get("default_account") or config.extra.get("account_id") or "")
        if not self._default_account:
            self._default_account = next(iter(self.accounts), "")
        self._retry_pending = pending
        self._retry_after = time.monotonic() + 30
        return not pending

    def _account_busy(self, account_id, child):
        return bool(child._active_sessions or child._pending_text_batches or child._send_text_gate.locked()
                    or self._outbound_inflight.get(account_id))

    def _publish_transport_config(self, config):
        runner = self.gateway_runner
        if runner is None:
            return
        owner = self._owner_profile
        runtime = (getattr(runner, "_profile_configs", None) or {}).get(owner) if owner else getattr(runner, "config", None)
        if runtime is not None:
            runtime.platforms[Platform.WEIXIN] = config

    def _config_signature(self):
        paths = [self._home / "config.yaml", self._home / ".env"]
        directory = self._home / "weixin" / "accounts"
        paths.extend(sorted(directory.glob("*.json")))
        result = []
        for path in paths:
            if path.name.endswith((".context-tokens.json", ".sync.json")):
                continue
            try:
                result.append((str(path), file_signature(path.stat())))
            except FileNotFoundError:
                result.append((str(path), None))
        return tuple(result)

    async def _reload_config(self):
        from gateway.run import _async_profile_runtime_scope
        from gateway.config import load_gateway_config
        from gateway.config_loader import read_yaml_layers
        # Includes new .env contents without mutating process secrets or agent prompts.
        async with _async_profile_runtime_scope(self._home):
            # The normal boot loader falls back on parse errors; hot reload must retain the live config.
            await asyncio.to_thread(read_yaml_layers, self._home)
            config = load_gateway_config().platforms.get(Platform.WEIXIN, PlatformConfig())
            for key in ("group_sessions_per_user", "thread_sessions_per_user"):
                config.extra.setdefault(key, self.config.extra.get(key, key == "group_sessions_per_user"))
            return await self.apply_config(config, is_reconnect=True)

    async def _watch_config(self):
        while True:
            await asyncio.sleep(2)
            if not self._home.is_dir():
                return
            try:
                signature = self._config_signature()
                retry = self._retry_pending and time.monotonic() >= self._retry_after
                if signature == self._signature and not retry:
                    continue
                if signature != self._pending_signature:
                    self._pending_signature = signature
                    continue  # Wait for setup's atomic writes to settle.
                if await self._reload_config():
                    self._signature = signature
                    logger.info("Weixin transport configuration applied (%d accounts)", len(self.accounts))
            except Exception:
                logger.warning("Weixin configuration reload failed; retrying", exc_info=True)

    async def send(self, chat_id, content, **kwargs):
        account_id = (kwargs.get("metadata") or {}).get("account_id")
        return await self._account_send(account_id, "send", chat_id, content, **kwargs)

    async def _account_send(self, account_id, method, *args, **kwargs):
        account_id = account_id or self._default_account
        child = self.accounts.get(account_id)
        if child is None:
            return SendResult(success=False, error="Requested Weixin account is offline")
        self._outbound_inflight[account_id] = self._outbound_inflight.get(account_id, 0) + 1
        try:
            return await getattr(child, method)(*args, **kwargs)
        finally:
            self._outbound_inflight[account_id] -= 1

    async def _send_media(self, method, chat_id, path, caption=None, **kwargs):
        account_id = (kwargs.get("metadata") or {}).get("account_id")
        return await self._account_send(account_id, method, chat_id, path, caption=caption, **kwargs)

    async def send_image(self, chat_id, image_url, caption="", **kwargs):
        return await self._send_media("send_image", chat_id, image_url, caption, **kwargs)

    async def send_image_file(self, chat_id, image_path, caption=None, **kwargs):
        return await self._send_media("send_image_file", chat_id, image_path, caption, **kwargs)

    async def send_document(self, chat_id, file_path, caption=None, **kwargs):
        return await self._send_media("send_document", chat_id, file_path, caption, **kwargs)

    async def send_video(self, chat_id, video_path, caption=None, **kwargs):
        return await self._send_media("send_video", chat_id, video_path, caption, **kwargs)

    async def send_voice(self, chat_id, audio_path, caption=None, **kwargs):
        return await self._send_media("send_voice", chat_id, audio_path, caption, **kwargs)

    async def edit_message(self, chat_id, message_id, content, **kwargs):
        return SendResult(success=False, error="Weixin does not support message editing")

    async def get_chat_info(self, chat_id):
        return {"chat_id": chat_id, "name": chat_id, "type": "dm"}

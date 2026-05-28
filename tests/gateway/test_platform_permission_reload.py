from types import SimpleNamespace

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.permissions import PlatformPermissionSnapshot


class _Adapter(BasePlatformAdapter):
    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {}


def test_base_adapter_permission_config_hook_replaces_config():
    old_config = PlatformConfig(enabled=True, extra={"allowed_chats": ["1"]})
    new_config = PlatformConfig(enabled=True, extra={"allowed_chats": ["2"]})
    adapter = _Adapter(old_config, Platform.TELEGRAM)

    BasePlatformAdapter.apply_permission_config(adapter, new_config, None)

    assert adapter.config is new_config


def test_telegram_adapter_permission_reload_recompiles_mentions_without_reconnect():
    from gateway.platforms.telegram import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    old_config = PlatformConfig(enabled=True, token="test", extra={"mention_patterns": ["rei"]})
    new_config = PlatformConfig(
        enabled=True,
        token="test",
        extra={"mention_patterns": ["ayanami"], "allowed_chats": ["-100"]},
    )
    adapter.platform = Platform.TELEGRAM
    adapter.config = old_config
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    adapter._mention_patterns = adapter._compile_mention_patterns()
    sentinel = object()
    adapter._application = sentinel

    permissions = PlatformPermissionSnapshot(
        platform=Platform.TELEGRAM,
        allowed_chats=["-100"],
        mention_patterns=["ayanami"],
    )
    adapter.apply_permission_config(new_config, permissions)

    assert adapter.config is new_config
    assert adapter._telegram_allowed_chats() == {"-100"}
    assert not adapter._mention_patterns[0].search("rei")
    assert adapter._mention_patterns[0].search("ayanami")
    assert adapter._application is sentinel

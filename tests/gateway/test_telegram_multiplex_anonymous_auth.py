from types import SimpleNamespace
import weakref

from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.run import GatewayRunner
from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


class Runner(GatewayRunner):
    def __init__(self, default_adapter, iot_adapter):
        self.adapters = {Platform.TELEGRAM: default_adapter}
        self._profile_adapters = {"iot-pm": {Platform.TELEGRAM: iot_adapter}}
        self._primary_profile_name = "default"
        self.config = SimpleNamespace(multiplex_profiles=True, platforms={})


class Adapter:
    pass


def _adapter(group_allowed_chats):
    a = Adapter()
    a.platform = Platform.TELEGRAM
    a.config = PlatformConfig(extra={"group_allowed_chats": group_allowed_chats})
    return a


def test_anonymous_group_uses_transport_allowlist_even_when_routed_profile_differs():
    default = _adapter(["-1004489232117"])
    iot = _adapter([])
    runner = Runner(default, iot)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="-1004489232117", chat_type="group", user_id=None,
        profile="iot-pm",
    )
    source._transport_adapter_ref = weakref.ref(default)
    observed = TelegramAdapter.__new__(TelegramAdapter)._telegram_group_observe_shared_source(source)
    assert runner._is_user_authorized_for_source(observed) is True

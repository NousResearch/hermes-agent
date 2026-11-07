"""Routing rows of a plugin platform survive a gateway start that loads them before the plugin
adapter registers its platform."""
from gateway.config import GatewayConfig, Platform
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.session import SessionSource, SessionStore

NAME = "routelate"


def _unregister():
    # A fresh gateway process: neither the registry entry nor the cached pseudo member exists.
    platform_registry.unregister(NAME)
    Platform._value2member_map_.pop(NAME, None)
    Platform._member_map_.pop(NAME.upper(), None)


def _register():
    platform_registry.register(PlatformEntry(
        name=NAME, label="Route Late", adapter_factory=lambda cfg: None, check_fn=lambda: True,
        source="plugin"))


def test_plugin_platform_route_survives_load_before_registration(tmp_path):
    _register()
    try:
        source = SessionSource(platform=Platform(NAME), chat_id="late-chat", user_id="late-user")
        first = SessionStore(tmp_path / "sessions", GatewayConfig())
        old = first.get_or_create_session(source)
        old.model_override = {"model": "late-model"}
        first._save()
        first._db.close()
    finally:
        _unregister()

    # Gateway start: the index loads before the plugin registers, and another platform saves.
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="other", user_id="u"))
    store._save()
    assert old.session_key in store._db.load_gateway_routing_entries(scope=store._routing_scope())

    _register()
    try:
        routed = store.get_or_create_session(source)
        assert routed.session_id == old.session_id
        # The stored entry itself is back, not a minimal one rebuilt from state.db.
        assert routed.model_override == {"model": "late-model"}
    finally:
        _unregister()
        store._db.close()

"""Yuanbao ships as a bundled platform plugin: registry entry shape + no core wiring left."""

from gateway.config import Platform


class _Ctx:
    def __init__(self):
        self.kwargs = None

    def register_platform(self, **kwargs):
        self.kwargs = kwargs


def test_register_shape_and_core_wiring_removed():
    from plugins.platforms.yuanbao import register
    from plugins.platforms.yuanbao.adapter import YuanbaoAdapter
    from plugins.platforms.yuanbao.register import _standalone_send

    ctx = _Ctx()
    register(ctx)
    kw = ctx.kwargs
    assert kw["name"] == "yuanbao" and kw["label"] == "Yuanbao" and kw["emoji"] == "💎"
    assert kw["adapter_factory"] is YuanbaoAdapter
    assert kw["standalone_sender_fn"] is _standalone_send
    assert kw["required_env"] == ["YUANBAO_APP_ID", "YUANBAO_APP_SECRET"]
    assert kw["cron_deliver_env_var"] == "YUANBAO_HOME_CHANNEL"
    assert kw["install_hint"] == "Yuanbao: websockets not installed. Run: pip install websockets"
    # /update was never allowed from Yuanbao (absent from _UPDATE_ALLOWED_PLATFORMS).
    assert kw["allow_update_command"] is False
    assert callable(kw["setup_fn"]) and callable(kw["is_connected"])

    from gateway.run import _BUILTIN_ADAPTERS
    from hermes_cli.gateway_setup_wizard import _PLATFORMS
    from tools import send_message_tool as smt
    assert Platform.YUANBAO not in _BUILTIN_ADAPTERS
    assert all(p["key"] != "yuanbao" for p in _PLATFORMS)
    assert "yuanbao" not in smt._CHUNKED_ROUTES
    # Media sends bypass the live adapter for the standalone sender (was a _PLUGIN_STANDALONE_MEDIA row).
    assert kw["standalone_send"].media is True

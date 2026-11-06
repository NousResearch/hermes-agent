"""The native card mode is opt-in and keeps the platform registration contract."""

from types import SimpleNamespace

import pytest

pytest.importorskip("lark_oapi")
pytest.importorskip("aiohttp")

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter, register


def test_factory_defaults_to_original_adapter_and_opt_in_uses_cards():
    registrations, hooks = {}, {}
    register(SimpleNamespace(
        register_platform=lambda **kw: registrations.update(kw),
        register_hook=lambda name, callback: hooks.update({name: callback}),
    ))
    factory = registrations["adapter_factory"]
    assert type(factory(PlatformConfig())) is FeishuAdapter
    assert type(factory(PlatformConfig(extra={"native_cards": "false"}))) is FeishuAdapter
    enabled = factory(PlatformConfig(extra={"native_cards": True}))
    assert isinstance(enabled, FeishuAdapter)
    assert type(enabled) is not FeishuAdapter
    assert enabled.supports_draft_streaming()
    assert set(hooks) == {
        "pre_tool_call", "post_tool_call", "pre_approval_request", "pre_api_request",
        "pre_auxiliary_call", "post_auxiliary_call",
    }
    assert registrations["env_overrides"]
    assert registrations["allowed_users_env"]
    assert registrations["standalone_sender_fn"]

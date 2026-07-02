"""register() metadata tests for the Wechaty platform plugin."""
from __future__ import annotations

from unittest.mock import MagicMock

from plugins.platforms.wechaty.adapter import register


def test_register_calls_register_platform() -> None:
    ctx = MagicMock()
    register(ctx)
    ctx.register_platform.assert_called_once()
    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "wechaty"
    assert kwargs["label"] == "WeChat (Wechaty)"
    assert kwargs["cron_deliver_env_var"] == "WECHATY_HOME_CHANNEL"
    assert kwargs["allowed_users_env"] == "WECHATY_ALLOWED_USERS"
    assert kwargs["allow_all_env"] == "WECHATY_ALLOW_ALL_USERS"
    assert kwargs["pii_safe"] is True
    assert kwargs["allow_update_command"] is True
    assert kwargs["max_message_length"] == 4000
    assert kwargs["emoji"] == "💬"
    assert callable(kwargs["check_fn"])
    assert callable(kwargs["validate_config"])
    assert callable(kwargs["is_connected"])
    assert callable(kwargs["env_enablement_fn"])
    assert callable(kwargs["apply_yaml_config_fn"])
    assert callable(kwargs["standalone_sender_fn"])
    assert callable(kwargs["setup_fn"])
    assert callable(kwargs["adapter_factory"])

import inspect
from unittest.mock import Mock

import pytest

from gateway.config import PlatformConfig


def test_registers_twitter_platform():
    from plugins.platforms.twitter import register

    ctx = Mock()
    register(ctx)

    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "twitter"
    assert kwargs["label"] == "Twitter / X"
    assert kwargs["allowed_users_env"] == "TWITTER_ALLOWED_USERS"
    assert kwargs["allow_all_env"] == "TWITTER_ALLOW_ALL_USERS"
    assert kwargs["cron_deliver_env_var"] == "TWITTER_HOME_CHANNEL"
    assert kwargs["max_message_length"] == 280
    assert callable(kwargs["standalone_sender_fn"])


def test_settings_reject_unsafe_limits():
    from plugins.platforms.twitter.adapter import TwitterSettings

    with pytest.raises(ValueError, match="poll_interval_seconds"):
        TwitterSettings.from_config(
            PlatformConfig(
                extra={"client_id": "client", "poll_interval_seconds": 0}
            )
        )


def test_adapter_send_signature_matches_base():
    from gateway.platforms.base import BasePlatformAdapter
    from plugins.platforms.twitter.adapter import TwitterAdapter

    assert inspect.signature(TwitterAdapter.send) == inspect.signature(
        BasePlatformAdapter.send
    )

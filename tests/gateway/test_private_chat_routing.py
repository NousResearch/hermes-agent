"""Exact owner-private routing for dedicated Telegram bot processes."""

import pytest

from gateway.private_chat_routing import PrivateChatRouteRegistry
from gateway.topic_routing import RouteDenied


def config(**overrides):
    raw = {
        "mode": "strict",
        "chat_id": 208214988,
        "user_id": 208214988,
        "profile": "atlas",
        "expected_bot_id": 900001,
        "expected_bot_username": "atlas_private_bot",
        "hooks": [{"profile": "atlas", "plugin": "sol_food"}],
    }
    raw.update(overrides)
    return raw


def test_exact_private_owner_resolves_without_synthetic_thread():
    registry = PrivateChatRouteRegistry.from_config(config())
    route = registry.resolve(
        chat_id=208214988,
        chat_type="private",
        user_id=208214988,
        thread_id=None,
    )
    assert route.profile == "atlas"
    assert route.thread_id is None


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"chat_id": 999}, "private_route_foreign_chat"),
        ({"user_id": 999}, "private_route_foreign_user"),
        ({"chat_type": "group"}, "private_route_not_private"),
        ({"thread_id": 1}, "private_route_unexpected_thread"),
    ],
)
def test_foreign_or_threaded_origin_fails_closed(changes, reason):
    kwargs = {
        "chat_id": 208214988,
        "chat_type": "private",
        "user_id": 208214988,
        "thread_id": None,
    }
    kwargs.update(changes)
    registry = PrivateChatRouteRegistry.from_config(config())
    with pytest.raises(RouteDenied) as caught:
        registry.resolve(**kwargs)
    assert caught.value.reason_code == reason


def test_outbound_is_exact_chat_and_has_no_thread():
    registry = PrivateChatRouteRegistry.from_config(config())
    assert registry.resolve_destination(chat_id=208214988, thread_id=None).profile == "atlas"
    with pytest.raises(RouteDenied):
        registry.resolve_destination(chat_id=999, thread_id=None)
    with pytest.raises(RouteDenied):
        registry.resolve_destination(chat_id=208214988, thread_id=1)


def test_bot_identity_seal_rejects_swapped_token():
    registry = PrivateChatRouteRegistry.from_config(config())
    registry.validate_bot_identity(bot_id=900001, username="Atlas_Private_Bot")
    with pytest.raises(ValueError, match="bot id"):
        registry.validate_bot_identity(bot_id=900002, username="atlas_private_bot")
    with pytest.raises(ValueError, match="username"):
        registry.validate_bot_identity(bot_id=900001, username="metis_private_bot")


@pytest.mark.parametrize(
    "bad",
    [
        {"mode": "fallback"},
        {"expected_bot_username": "@atlas_private_bot"},
        {"expected_bot_username": "Atlas_Private_Bot"},
        {"expected_bot_id": "not-an-id"},
        {"extra": "forbidden"},
    ],
)
def test_config_shape_is_closed(bad):
    with pytest.raises(ValueError):
        PrivateChatRouteRegistry.from_config(config(**bad))


def test_gateway_config_accepts_one_dedicated_private_route():
    from gateway.config import GatewayConfig, Platform

    gateway = GatewayConfig.from_dict(
        {
            "platforms": {
                "telegram": {
                    "enabled": True,
                    "token": "synthetic-test-token",
                    "extra": {"private_chat_routing": config()},
                }
            }
        }
    )
    assert gateway.platforms[Platform.TELEGRAM].extra["private_chat_routing"][
        "profile"
    ] == "atlas"


def test_gateway_config_rejects_private_route_with_dm_topics():
    from gateway.config import GatewayConfig

    with pytest.raises(ValueError, match="cannot combine with dm_topics"):
        GatewayConfig.from_dict(
            {
                "platforms": {
                    "telegram": {
                        "enabled": True,
                        "token": "synthetic-test-token",
                        "extra": {
                            "private_chat_routing": config(),
                            "dm_topics": [
                                {"chat_id": 208214988, "topics": [{"name": "Old"}]}
                            ],
                        },
                    }
                }
            }
        )


def test_gateway_config_rejects_private_and_topic_modes_together():
    from gateway.config import GatewayConfig

    with pytest.raises(ValueError, match="cannot combine"):
        GatewayConfig.from_dict(
            {
                "multiplex_profiles": True,
                "profile_routes": [
                    {
                        "name": "atlas-topic",
                        "platform": "telegram",
                        "chat_id": "208214988",
                        "thread_id": "1",
                        "profile": "atlas",
                    }
                ],
                "platforms": {
                    "telegram": {
                        "enabled": True,
                        "token": "synthetic-test-token",
                        "extra": {
                            "private_chat_routing": config(),
                            "topic_routing": {
                                "mode": "strict",
                                "routes": [
                                    {
                                        "chat_id": "208214988",
                                        "thread_id": 1,
                                        "profile": "atlas",
                                    }
                                ],
                            },
                        },
                    }
                },
            }
        )

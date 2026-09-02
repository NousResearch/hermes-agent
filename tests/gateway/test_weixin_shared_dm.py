"""Shared Weixin DM session behavior."""

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.weixin import WeixinAdapter, WeixinMultiAccountAdapter
from gateway.session import (
    SessionSource,
    build_session_key,
    is_shared_multi_user_session,
)


def _weixin_dm(user_id: str, *, shared_session_id: str | None = None) -> SessionSource:
    return SessionSource(
        platform=Platform.WEIXIN,
        chat_id=user_id,
        chat_type="dm",
        user_id=user_id,
        user_name=user_id,
        shared_session_id=shared_session_id,
    )


def test_configured_shared_weixin_dms_use_one_session_key():
    boge = _weixin_dm("boge", shared_session_id="boge-huihui")
    huihui = _weixin_dm("huihui", shared_session_id="boge-huihui")

    assert build_session_key(boge) == build_session_key(huihui)
    assert build_session_key(boge) == "agent:main:weixin:dm:shared:boge-huihui"
    assert is_shared_multi_user_session(boge)


def test_unconfigured_weixin_dm_remains_isolated():
    boge = _weixin_dm("boge")
    huihui = _weixin_dm("huihui")

    assert build_session_key(boge) != build_session_key(huihui)
    assert not is_shared_multi_user_session(boge)


def test_non_weixin_shared_marker_is_ignored():
    telegram = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="telegram-user",
        chat_type="dm",
        user_id="telegram-user",
        shared_session_id="boge-huihui",
    )

    assert not is_shared_multi_user_session(telegram)
    assert build_session_key(telegram) == "agent:main:telegram:dm:telegram-user"


def test_shared_session_identity_round_trips_through_source_dict():
    source = _weixin_dm("boge", shared_session_id="boge-huihui")
    source.transport_account_id = "boge@im.bot"

    restored = SessionSource.from_dict(source.to_dict())

    assert restored.shared_session_id == "boge-huihui"
    assert restored.transport_account_id == "boge@im.bot"
    assert build_session_key(restored) == build_session_key(source)


def test_weixin_adapter_only_marks_configured_users_for_sharing():
    adapter = WeixinAdapter(
        PlatformConfig(
            extra={
                "account_id": "bot@im.bot",
                "shared_dm_session": "boge-huihui",
                "shared_dm_users": ["boge", "huihui"],
                "shared_dm_sender_name": "大哥",
            }
        )
    )

    assert adapter._shared_session_id_for_sender("boge") == "boge-huihui"
    assert adapter._shared_session_id_for_sender("huihui") == "boge-huihui"
    assert adapter._shared_session_id_for_sender("stranger") is None
    assert adapter._display_name_for_sender("boge") == "大哥"
    assert adapter._display_name_for_sender("stranger") == "stranger"


def test_multi_account_adapter_builds_isolated_ilink_children():
    adapter = WeixinMultiAccountAdapter(
        PlatformConfig(
            extra={
                "accounts": [
                    {
                        "name": "boge",
                        "account_id": "boge@im.bot",
                        "token": "token-boge",
                        "shared_dm_session": "boge-huihui",
                        "shared_dm_users": ["boge-user"],
                    },
                    {
                        "name": "huihui",
                        "account_id": "huihui@im.bot",
                        "token": "token-huihui",
                        "shared_dm_session": "boge-huihui",
                        "shared_dm_users": ["huihui-user"],
                    },
                ]
            }
        )
    )

    assert [child._account_id for child in adapter._children] == [
        "boge@im.bot",
        "huihui@im.bot",
    ]
    assert adapter._children[0]._token == "token-boge"
    assert adapter._children[1]._token == "token-huihui"
    assert adapter._children[0]._shared_session_id_for_sender("boge-user") == "boge-huihui"
    assert adapter._children[1]._shared_session_id_for_sender("huihui-user") == "boge-huihui"


def test_gateway_recognizes_saved_multi_account_weixin_config():
    config = GatewayConfig.from_dict(
        {
            "platforms": {
                "weixin": {
                    "enabled": True,
                    "extra": {
                        "accounts": [
                            {"account_id": "a@im.bot", "token": "token-a"},
                            {"account_id": "b@im.bot", "token": "token-b"},
                        ]
                    },
                }
            }
        }
    )

    platform_config = config.platforms[Platform.WEIXIN]
    assert config._is_platform_connected(Platform.WEIXIN, platform_config)

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


def _config_with_profiles():
    return GatewayConfig(
        platforms={
            Platform.FEISHU: PlatformConfig(
                enabled=True,
                extra={
                    "sender_profiles": {
                        "ou_owner": {
                            "name": "孙炜臻",
                            "role": "owner",
                            "relationship": "主人 / LangLang owner",
                            "address_as": "主人",
                        },
                        "ou_collab": {
                            "name": "小鱼",
                            "role": "collaborator",
                            "relationship": "工作群协作方，不是主人",
                            "address_as": "小鱼",
                            "not_owner": True,
                        },
                    }
                },
            )
        },
        group_sessions_per_user=False,
    )


def test_shared_sender_prefix_uses_configured_owner_relationship():
    from gateway.session import format_shared_sender_prefix

    cfg = _config_with_profiles()
    src = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_workgroup",
        chat_type="group",
        user_id="ou_owner",
        user_name="Raw Owner Name",
    )

    assert format_shared_sender_prefix(src, cfg) == (
        "[Speaker: 孙炜臻 | role: owner | relationship: 主人 / LangLang owner | address_as: 主人]"
    )


def test_shared_sender_prefix_marks_configured_non_owner_collaborator():
    from gateway.session import format_shared_sender_prefix

    cfg = _config_with_profiles()
    src = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_workgroup",
        chat_type="group",
        user_id="ou_collab",
        user_name="Raw Collab Name",
    )

    assert format_shared_sender_prefix(src, cfg) == (
        "[Speaker: 小鱼 | role: collaborator | relationship: 工作群协作方，不是主人 | address_as: 小鱼 | not_owner: true]"
    )


def test_shared_sender_prefix_unknown_sender_has_safe_non_owner_fallback():
    from gateway.session import format_shared_sender_prefix

    cfg = _config_with_profiles()
    src = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_workgroup",
        chat_type="group",
        user_id="ou_unknown",
        user_name=None,
    )

    assert format_shared_sender_prefix(src, cfg) == (
        "[Speaker: unknown Feishu user ou_unknown | role: unknown | address_as: 对方/这位用户 | not_owner: true]"
    )

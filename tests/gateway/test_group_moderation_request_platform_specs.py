from gateway.config import Platform
from gateway.group_moderation_request_platform_specs import (
    build_qq_group_moderation_request_platform_spec,
    build_weixin_group_moderation_request_platform_spec,
)


def test_build_qq_group_moderation_request_platform_spec_uses_injected_overrides():
    spec = build_qq_group_moderation_request_platform_spec(
        request_matcher=lambda **kwargs: ({"action": "kick_user"}, None),
        action_matcher=lambda body: "kick_user",
        user_query_extractor=lambda body: "广告哥",
        reason_extractor=lambda body: "广告",
        duration_extractor=lambda body: 600,
    )

    assert spec.platform is Platform.QQ_NAPCAT
    assert spec.request_matcher(body="x", source=None, admin_ids_configured=True, is_admin_user=True, admin_only_message="x")[0] == {
        "action": "kick_user"
    }
    assert spec.action_matcher("把他踢了") == "kick_user"
    assert spec.user_query_extractor("把广告哥踢了") == "广告哥"
    assert spec.reason_extractor("原因广告") == "广告"
    assert spec.duration_extractor("10 分钟") == 600


def test_build_weixin_group_moderation_request_platform_spec_uses_injected_overrides():
    spec = build_weixin_group_moderation_request_platform_spec(
        request_matcher=lambda **kwargs: ({"action": "mute_user"}, None),
        action_matcher=lambda body: "mute_user",
        user_query_extractor=lambda body: "卖草的",
        reason_extractor=lambda body: "广告",
        duration_extractor=lambda body: 300,
    )

    assert spec.platform is Platform.WEIXIN
    assert spec.request_matcher(body="x", source=None, admin_ids_configured=True, is_admin_user=True, admin_only_message="x")[0] == {
        "action": "mute_user"
    }
    assert spec.action_matcher("禁言他") == "mute_user"
    assert spec.user_query_extractor("把卖草的禁言") == "卖草的"
    assert spec.reason_extractor("原因广告") == "广告"
    assert spec.duration_extractor("5 分钟") == 300

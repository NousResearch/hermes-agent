from gateway.config import Platform
from gateway.group_visible_addressing_platform_specs import (
    build_qq_group_visible_addressing_platform_spec,
    build_weixin_group_visible_addressing_platform_spec,
)


def test_build_qq_group_visible_addressing_platform_spec_uses_injected_aliases():
    spec = build_qq_group_visible_addressing_platform_spec(
        visible_name_aliases=("马嘎", "马噶"),
    )

    assert spec.platform is Platform.QQ_NAPCAT
    assert spec.has_visible_bot_address("@马嘎 继续处理") is True
    assert spec.has_visible_bot_address("继续处理") is False


def test_build_weixin_group_visible_addressing_platform_spec_uses_injected_aliases():
    spec = build_weixin_group_visible_addressing_platform_spec(
        visible_name_aliases=("马嘎",),
    )

    assert spec.platform is Platform.WEIXIN
    assert spec.has_visible_bot_address("马嘎 看下这个群") is True
    assert spec.has_visible_bot_address("看下这个群") is False

"""Platform specs for visible bot-address detection in group chats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from gateway.config import Platform
from gateway.qq_intents import _QQ_VISIBLE_NAME_ALIASES


@dataclass(frozen=True)
class GroupVisibleAddressingPlatformSpec:
    platform: Platform
    has_visible_bot_address: Callable[[str], bool]


def _body_has_visible_alias(message_text: str, aliases: tuple[str, ...]) -> bool:
    body = str(message_text or "").strip()
    return bool(body) and any(alias in body for alias in aliases)


def build_qq_group_visible_addressing_platform_spec(
    *,
    visible_name_aliases: tuple[str, ...] | None = None,
) -> GroupVisibleAddressingPlatformSpec:
    aliases = tuple(visible_name_aliases or _QQ_VISIBLE_NAME_ALIASES)
    return GroupVisibleAddressingPlatformSpec(
        platform=Platform.QQ_NAPCAT,
        has_visible_bot_address=lambda body: _body_has_visible_alias(body, aliases),
    )


def build_weixin_group_visible_addressing_platform_spec(
    *,
    visible_name_aliases: tuple[str, ...] | None = None,
) -> GroupVisibleAddressingPlatformSpec:
    aliases = tuple(visible_name_aliases or _QQ_VISIBLE_NAME_ALIASES)
    return GroupVisibleAddressingPlatformSpec(
        platform=Platform.WEIXIN,
        has_visible_bot_address=lambda body: _body_has_visible_alias(body, aliases),
    )


QQ_GROUP_VISIBLE_ADDRESSING_PLATFORM_SPEC = build_qq_group_visible_addressing_platform_spec()
WEIXIN_GROUP_VISIBLE_ADDRESSING_PLATFORM_SPEC = build_weixin_group_visible_addressing_platform_spec()


def get_group_visible_addressing_platform_spec(platform: Platform | None) -> GroupVisibleAddressingPlatformSpec:
    if platform is Platform.WEIXIN:
        return WEIXIN_GROUP_VISIBLE_ADDRESSING_PLATFORM_SPEC
    return QQ_GROUP_VISIBLE_ADDRESSING_PLATFORM_SPEC

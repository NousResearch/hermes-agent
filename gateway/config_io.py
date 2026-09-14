"""Home configuration persistence and provenance across delivery overlays."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.config import GatewayConfig, HomeChannel, Platform


def snapshot_home_bindings(config: GatewayConfig) -> dict[Platform, HomeChannel]:
    return {
        platform: replace(platform_config.home_channel)
        for platform, platform_config in config.platforms.items()
        if platform_config.home_channel is not None
    }


def restore_matching_home_bindings(
    config: GatewayConfig, previous: dict[Platform, HomeChannel]
) -> None:
    for platform, selected in previous.items():
        current = config.get_home_channel(platform)
        if (
            current is None
            or current.platform != selected.platform
            or str(current.chat_id) != str(selected.chat_id)
            or str(current.thread_id or "") != str(selected.thread_id or "")
        ):
            continue
        # An explicit replacement wins; a different destination inherits no provenance.
        if current.user_id and str(current.user_id) != str(selected.user_id or ""):
            continue
        if current.scope_id and str(current.scope_id) != str(selected.scope_id or ""):
            continue
        current.user_id = selected.user_id
        current.scope_id = selected.scope_id
        if current.name == "Home":
            current.name = selected.name
        current.selection_id = selected.selection_id
        current.group_audience_ack = selected.group_audience_ack


def persist_home(home: HomeChannel, *, enabled_if_new: bool = False) -> None:
    from hermes_cli.config import _CONFIG_LOCK, load_config, save_config

    with _CONFIG_LOCK:
        config = load_config()
        platforms = config.setdefault("platforms", {})
        if not isinstance(platforms, dict):
            platforms = config["platforms"] = {}
        platform_config = platforms.setdefault(home.platform.value, {})
        if not isinstance(platform_config, dict):
            platform_config = platforms[home.platform.value] = {}
        if enabled_if_new:
            platform_config.setdefault("enabled", True)
        platform_config["home_channel"] = home.to_dict()
        save_config(config)

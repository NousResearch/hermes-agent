"""Discovery helpers for Tlon groups and channels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Set


@dataclass
class TlonDiscovery:
    channels: Set[str] = field(default_factory=set)
    channel_to_group: Dict[str, str] = field(default_factory=dict)
    group_names: Dict[str, str] = field(default_factory=dict)
    foreigns: Dict[str, Any] = field(default_factory=dict)


def parse_groups_ui_init(raw: Any) -> TlonDiscovery:
    """Parse ``/groups-ui/v7/init.json`` style data."""
    result = TlonDiscovery()
    if not isinstance(raw, dict):
        return result

    groups = raw.get("groups")
    if isinstance(groups, dict):
        for group_flag, group_data in groups.items():
            if not isinstance(group_flag, str) or not isinstance(group_data, dict):
                continue
            title = (group_data.get("meta") or {}).get("title")
            if isinstance(title, str) and title:
                result.group_names[group_flag] = title

            channels = group_data.get("channels")
            if not isinstance(channels, dict):
                continue
            for nest in channels.keys():
                if _is_tlon_channel(nest):
                    result.channels.add(nest)
                    result.channel_to_group[nest] = group_flag

    foreigns = raw.get("foreigns")
    if isinstance(foreigns, dict):
        result.foreigns = foreigns

    return result


def parse_legacy_groups(raw: Any) -> TlonDiscovery:
    """Parse older ``/groups/v1/groups.json`` style data."""
    result = TlonDiscovery()
    if not isinstance(raw, dict):
        return result

    for group_flag, group_data in raw.items():
        if not isinstance(group_flag, str) or not isinstance(group_data, dict):
            continue
        meta = group_data.get("meta") or {}
        title = meta.get("title") if isinstance(meta, dict) else None
        if isinstance(title, str) and title:
            result.group_names[group_flag] = title

        channels = group_data.get("channels")
        if not isinstance(channels, dict):
            continue
        for nest in channels.keys():
            if _is_tlon_channel(nest):
                result.channels.add(nest)
                result.channel_to_group[nest] = group_flag

    return result


def pending_group_invites(foreigns: Dict[str, Any]) -> list[dict[str, Any]]:
    """Return valid pending foreign group invites from groups-ui foreign data."""
    invites: list[dict[str, Any]] = []
    for group_flag, data in foreigns.items():
        if not isinstance(data, dict):
            continue
        raw_invites = data.get("invites")
        if not isinstance(raw_invites, list):
            continue
        for invite in raw_invites:
            if isinstance(invite, dict) and invite.get("valid"):
                item = dict(invite)
                item["groupFlag"] = group_flag
                preview = data.get("preview")
                if isinstance(preview, dict):
                    item["groupTitle"] = (
                        ((preview.get("meta") or {}).get("title"))
                        if isinstance(preview.get("meta"), dict)
                        else None
                    )
                invites.append(item)
    return invites


def _is_tlon_channel(nest: Any) -> bool:
    return isinstance(nest, str) and nest.startswith(("chat/", "heap/", "diary/"))

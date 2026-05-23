"""Settings-store parsing for the Hermes Tlon adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SETTINGS_DESK = "moltbot"
SETTINGS_BUCKET = "tlon"


@dataclass
class TlonSettings:
    group_channels: Optional[List[str]] = None
    dm_allowlist: Optional[List[str]] = None
    auto_discover: Optional[bool] = None
    auto_accept_dm_invites: Optional[bool] = None
    auto_accept_group_invites: Optional[bool] = None
    group_invite_allowlist: Optional[List[str]] = None
    channel_rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_authorized_ships: Optional[List[str]] = None
    owner_ship: Optional[str] = None
    pending_approvals: Optional[List[Dict[str, Any]]] = None
    show_model_signature: Optional[bool] = None
    owner_listen_enabled: Optional[bool] = None
    owner_listen_disabled_channels: Optional[List[str]] = None


def parse_settings_response(raw: Any) -> TlonSettings:
    """Parse ``/settings/all.json`` response for desk ``moltbot`` bucket ``tlon``."""
    bucket = _extract_bucket(raw)
    settings = TlonSettings()
    if not bucket:
        return settings

    settings.group_channels = _string_list(bucket.get("groupChannels"))
    settings.dm_allowlist = _string_list(bucket.get("dmAllowlist"))
    settings.auto_discover = _bool_or_none(bucket.get("autoDiscover"))
    settings.auto_accept_dm_invites = _bool_or_none(bucket.get("autoAcceptDmInvites"))
    settings.auto_accept_group_invites = _bool_or_none(bucket.get("autoAcceptGroupInvites"))
    settings.group_invite_allowlist = _string_list(bucket.get("groupInviteAllowlist"))
    settings.channel_rules = _channel_rules(bucket.get("channelRules"))
    settings.default_authorized_ships = _string_list(bucket.get("defaultAuthorizedShips"))
    settings.owner_ship = bucket.get("ownerShip") if isinstance(bucket.get("ownerShip"), str) else None
    settings.pending_approvals = _pending_approvals(bucket.get("pendingApprovals"))
    settings.show_model_signature = _bool_or_none(bucket.get("showModelSig"))
    settings.owner_listen_enabled = _bool_or_none(bucket.get("ownerListenEnabled"))
    settings.owner_listen_disabled_channels = _string_list(bucket.get("ownerListenDisabledChannels"))
    return settings


def parse_settings_event(event: Any) -> Optional[tuple[str, Any]]:
    """Parse a settings-store subscription event into ``(key, value)``."""
    if not isinstance(event, dict):
        return None
    evt = event.get("settings-event")
    if isinstance(evt, dict):
        event = evt

    put = event.get("put-entry")
    if isinstance(put, dict):
        if put.get("desk") == SETTINGS_DESK and put.get("bucket-key") == SETTINGS_BUCKET:
            key = put.get("entry-key")
            return (str(key), put.get("value")) if key else None

    delete = event.get("del-entry")
    if isinstance(delete, dict):
        if delete.get("desk") == SETTINGS_DESK and delete.get("bucket-key") == SETTINGS_BUCKET:
            key = delete.get("entry-key")
            return (str(key), None) if key else None

    return None


def apply_settings_update(current: TlonSettings, key: str, value: Any) -> TlonSettings:
    """Return a copy of ``current`` with one settings-store update applied."""
    data = {
        "group_channels": current.group_channels,
        "dm_allowlist": current.dm_allowlist,
        "auto_discover": current.auto_discover,
        "auto_accept_dm_invites": current.auto_accept_dm_invites,
        "auto_accept_group_invites": current.auto_accept_group_invites,
        "group_invite_allowlist": current.group_invite_allowlist,
        "channel_rules": dict(current.channel_rules),
        "default_authorized_ships": current.default_authorized_ships,
        "owner_ship": current.owner_ship,
        "pending_approvals": current.pending_approvals,
        "show_model_signature": current.show_model_signature,
        "owner_listen_enabled": current.owner_listen_enabled,
        "owner_listen_disabled_channels": current.owner_listen_disabled_channels,
    }

    if key == "groupChannels":
        data["group_channels"] = _string_list(value)
    elif key == "dmAllowlist":
        data["dm_allowlist"] = _string_list(value)
    elif key == "autoDiscover":
        data["auto_discover"] = _bool_or_none(value)
    elif key == "autoAcceptDmInvites":
        data["auto_accept_dm_invites"] = _bool_or_none(value)
    elif key == "autoAcceptGroupInvites":
        data["auto_accept_group_invites"] = _bool_or_none(value)
    elif key == "groupInviteAllowlist":
        data["group_invite_allowlist"] = _string_list(value)
    elif key == "channelRules":
        data["channel_rules"] = _channel_rules(value)
    elif key == "defaultAuthorizedShips":
        data["default_authorized_ships"] = _string_list(value)
    elif key == "ownerShip":
        data["owner_ship"] = value if isinstance(value, str) else None
    elif key == "pendingApprovals":
        data["pending_approvals"] = _pending_approvals(value)
    elif key == "showModelSig":
        data["show_model_signature"] = _bool_or_none(value)
    elif key == "ownerListenEnabled":
        data["owner_listen_enabled"] = _bool_or_none(value)
    elif key == "ownerListenDisabledChannels":
        data["owner_listen_disabled_channels"] = _string_list(value)

    return TlonSettings(**data)


def _extract_bucket(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    all_data = raw.get("all")
    if isinstance(all_data, dict):
        desk = all_data.get(SETTINGS_DESK)
    else:
        desk = raw.get(SETTINGS_DESK) if SETTINGS_DESK in raw else raw
    if not isinstance(desk, dict):
        return {}
    bucket = desk.get(SETTINGS_BUCKET)
    return bucket if isinstance(bucket, dict) else {}


def _string_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, str)]


def _bool_or_none(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _channel_rules(value: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    if not isinstance(value, dict):
        return {}

    rules: Dict[str, Dict[str, Any]] = {}
    for channel, rule in value.items():
        if not isinstance(channel, str) or not isinstance(rule, dict):
            continue
        mode = rule.get("mode")
        ships = rule.get("allowedShips")
        parsed: Dict[str, Any] = {}
        if mode in {"open", "restricted"}:
            parsed["mode"] = mode
        if isinstance(ships, list):
            parsed["allowedShips"] = [s for s in ships if isinstance(s, str)]
        rules[channel] = parsed
    return rules


def _pending_approvals(value: Any) -> Optional[List[Dict[str, Any]]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if not isinstance(value, list):
        return None
    result: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            result.append(item)
    return result

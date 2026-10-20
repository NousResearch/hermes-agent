"""Bot Mode group chat CRUD — the backend for the ``hermes group`` subcommand.

Bot Mode group chats (the desktop Bots tab → group rooms) are backed by profile
metadata, not a database. Two ``ui_meta`` keys under ``profile.yaml``:

* each bot's own profile → ``ui_meta.hermes-bots.groups`` (its membership list)
* the default profile   → ``ui_meta.hermes-bots-groups`` (the room projection:
  durable roomId, members, cross-client mirror, rename/disband tombstones)

This module writes the *same contract* the desktop's Bots plugin writes through
the ``profiles.configure`` gateway RPC, so a group created here is
indistinguishable from one created in the UI — the desktop rehydrates rooms
from this projection on every cold start.
"""

from __future__ import annotations

import random
import string
import time
from typing import Dict, List, Optional, Set, Tuple

from hermes_cli.profiles import (
    get_profile_dir,
    list_profile_names,
    read_profile_ui_meta,
    set_profile_ui_meta,
)

BOT_META_KEY = "hermes-bots"
PROJECTION_KEY = "hermes-bots-groups"
MAX_MEMBERS = 6  # desktop GROUP_CHAT_MAX_MEMBERS
LOCAL_CONNECTION = {
    "connectionId": "local",
    "connectionKind": "local",
    "connectionLabel": "This device",
    "sourceScoped": True,
}


def _bot_meta(profile_dir) -> dict:
    meta = read_profile_ui_meta(profile_dir, BOT_META_KEY)
    return meta if isinstance(meta, dict) else {}


def _bot_groups(meta: dict) -> List[str]:
    """Group names a bot belongs to. ``groups`` is authoritative; the legacy
    ``group`` scalar is the fallback (single value, no comma-split)."""
    groups = meta.get("groups")
    if isinstance(groups, list):
        return [g for g in groups if isinstance(g, str) and g.strip()]
    scalar = meta.get("group")
    if isinstance(scalar, str) and scalar.strip():
        return [scalar.strip()]
    return []


def _set_bot_groups(bot: str, group: str, enabled: bool) -> None:
    """Add/remove one group from a bot's membership, preserving appearance."""
    profile_dir = get_profile_dir(bot)
    meta = dict(_bot_meta(profile_dir))
    groups = _bot_groups(meta)
    if enabled:
        if group and group not in groups:
            groups.append(group)
    else:
        groups = [g for g in groups if g != group]
    meta["groups"] = groups
    if groups:
        meta["group"] = groups[0]
    else:
        meta.pop("group", None)
    set_profile_ui_meta(profile_dir, BOT_META_KEY, meta)


def _projection() -> dict:
    proj = read_profile_ui_meta(get_profile_dir("default"), PROJECTION_KEY)
    if not isinstance(proj, dict):
        return {"version": 3, "updatedAt": 0, "rooms": {}, "deleted": {}}
    proj = dict(proj)
    proj.setdefault("version", 3)
    proj.setdefault("updatedAt", 0)
    proj.setdefault("rooms", {})
    proj.setdefault("deleted", {})
    return proj


def _save_projection(proj: dict) -> None:
    set_profile_ui_meta(get_profile_dir("default"), PROJECTION_KEY, proj)


def _mint_room_id() -> str:
    # Matches the desktop's mintGroupRoomId(): r<base36-ms>-<5 chars>.
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return f"r{_base36(int(time.time() * 1000))}-{suffix}"


def _base36(n: int) -> str:
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    if n == 0:
        return "0"
    out = ""
    while n:
        n, r = divmod(n, 36)
        out = chars[r] + out
    return out


def _member_descriptor(bot: str) -> dict:
    return {"name": bot, "handle": bot, **LOCAL_CONNECTION}


def _find_room(proj: dict, name: str) -> Tuple[Optional[str], Optional[dict]]:
    for key, room in proj.get("rooms", {}).items():
        if isinstance(room, dict) and room.get("name") == name:
            return key, room
    legacy = f"name:{name}"
    if legacy in proj.get("rooms", {}):
        return legacy, proj["rooms"][legacy]
    return None, None


def _next_revision(proj: dict) -> int:
    latest = max(
        [int(r.get("revision", 0) or 0) for r in proj.get("rooms", {}).values() if isinstance(r, dict)]
        or [0]
    )
    return latest + 1


def _touch(proj: dict) -> None:
    proj["updatedAt"] = int(time.time() * 1000)
    proj["version"] = 3


def _unique_name(base: str, taken: Set[str]) -> str:
    # Matches the desktop's uniqueGroupChatName(): truncate 64, " 2", " 3", …
    if base not in taken:
        return base
    for n in range(2, 100):
        suffix = f" {n}"
        candidate = base[: 64 - len(suffix)] + suffix
        if candidate not in taken:
            return candidate
    raise ValueError("no free name for the group")


def _clean_bots(bots: List[str]) -> List[str]:
    seen: List[str] = []
    for b in bots:
        b = b.strip()
        if b and b not in seen:
            seen.append(b)
    return seen


# ── public API ──────────────────────────────────────────────────────────────


def list_bots() -> List[str]:
    return list_profile_names()


def list_groups() -> Dict[str, dict]:
    """Union of bot-meta membership and the room projection.

    Returns {name: {"members": [sorted], "roomId": str|None}}."""
    groups: Dict[str, dict] = {}
    for bot in list_profile_names():
        for g in _bot_groups(_bot_meta(get_profile_dir(bot))):
            groups.setdefault(g, {"members": set(), "roomId": None})
            groups[g]["members"].add(bot)
    for key, room in _projection().get("rooms", {}).items():
        if not isinstance(room, dict) or room.get("tombstone"):
            continue
        name = room.get("name") or (key[5:] if key.startswith("name:") else key)
        groups.setdefault(name, {"members": set(), "roomId": None})
        rid = room.get("roomId")
        if isinstance(rid, str) and rid:
            groups[name]["roomId"] = rid
        for m in room.get("members", []):
            if isinstance(m, dict) and m.get("name"):
                groups[name]["members"].add(m["name"])
    return {
        n: {"members": sorted(m["members"]), "roomId": m["roomId"]}
        for n, m in sorted(groups.items())
    }


def create_group(name: str, bots: List[str]) -> dict:
    bots = _clean_bots(bots)
    if len(bots) < 2:
        raise ValueError("a group chat needs at least 2 bots")
    if len(bots) > MAX_MEMBERS:
        raise ValueError(f"a group chat holds at most {MAX_MEMBERS} bots (got {len(bots)})")
    name = _unique_name(name.strip()[:64], set(list_groups().keys()))
    for bot in bots:
        _set_bot_groups(bot, name, True)
    proj = _projection()
    room_id = _mint_room_id()
    proj.setdefault("rooms", {})[f"id:{room_id}"] = {
        "name": name,
        "roomId": room_id,
        "log": [],
        "members": [_member_descriptor(b) for b in bots],
        "revision": _next_revision(proj),
    }
    _touch(proj)
    _save_projection(proj)
    return {"name": name, "roomId": room_id, "members": bots}


def _update_projection_members(name: str, members: List[str]) -> None:
    proj = _projection()
    _, room = _find_room(proj, name)
    if room is not None:
        room["members"] = [_member_descriptor(b) for b in members]
        room["revision"] = _next_revision(proj)
        _touch(proj)
        _save_projection(proj)


def add_members(name: str, bots: List[str]) -> dict:
    bots = _clean_bots(bots)
    groups = list_groups()
    if name not in groups:
        raise ValueError(f"group '{name}' does not exist")
    current = set(groups[name]["members"])
    added = [b for b in bots if b not in current]
    if len(current | set(added)) > MAX_MEMBERS:
        raise ValueError(f"a group chat holds at most {MAX_MEMBERS} bots")
    for bot in added:
        _set_bot_groups(bot, name, True)
    if added:
        _update_projection_members(name, sorted(current | set(added)))
    return {"name": name, "added": added}


def remove_members(name: str, bots: List[str]) -> dict:
    bots = _clean_bots(bots)
    groups = list_groups()
    if name not in groups:
        raise ValueError(f"group '{name}' does not exist")
    current = set(groups[name]["members"])
    removed = [b for b in bots if b in current]
    for bot in removed:
        _set_bot_groups(bot, name, False)
    _update_projection_members(name, sorted(current - set(removed)))
    return {"name": name, "removed": removed}


def rename_group(old: str, new: str) -> dict:
    old, new = old.strip(), new.strip()
    if not new:
        raise ValueError("new name is required")
    groups = list_groups()
    if old not in groups:
        raise ValueError(f"group '{old}' does not exist")
    if new in groups:
        raise ValueError(f"group '{new}' already exists")
    for bot in groups[old]["members"]:
        _set_bot_groups(bot, old, False)
        _set_bot_groups(bot, new, True)
    proj = _projection()
    key, room = _find_room(proj, old)
    if room is not None:
        room["name"] = new
        room["revision"] = _next_revision(proj)
        if key.startswith("name:"):
            del proj["rooms"][key]
            proj["rooms"][f"name:{new}"] = room
        _touch(proj)
        _save_projection(proj)
    return {"old": old, "new": new}


def disband_group(name: str) -> dict:
    name = name.strip()
    groups = list_groups()
    if name not in groups:
        raise ValueError(f"group '{name}' does not exist")
    for bot in groups[name]["members"]:
        _set_bot_groups(bot, name, False)
    proj = _projection()
    key, _ = _find_room(proj, name)
    if key is not None:
        proj.setdefault("deleted", {})[key] = _next_revision(proj)
        proj.setdefault("rooms", {}).pop(key, None)
        _touch(proj)
        _save_projection(proj)
    return {"name": name}


def group_info(name: str) -> dict:
    groups = list_groups()
    if name not in groups:
        return {"name": name, "exists": False}
    info = {"name": name, "exists": True, **groups[name]}
    return info


# ── command handler ──────────────────────────────────────────────────────────


def _split(csv: str) -> List[str]:
    return _clean_bots(csv.split(","))


def cmd_group(args) -> None:
    """Dispatch ``hermes group <action>``."""
    action = getattr(args, "group_action", None)

    if action in (None, "list"):
        groups = list_groups()
        if not groups:
            print("(no groups)")
            return
        for name, info in groups.items():
            members = ", ".join(info["members"]) or "—"
            tag = f"  (id:{info['roomId']})" if info.get("roomId") else ""
            print(f"{name}  [{len(info['members'])}]{tag}\n    {members}")
        return

    if action == "bots":
        for b in list_bots():
            print(b)
        return

    if action == "create":
        res = create_group(args.name, _split(args.bots))
        print(f"created '{res['name']}' ({res['roomId']}) with {len(res['members'])} bots")
        return

    if action == "info":
        info = group_info(args.name)
        if not info["exists"]:
            print(f"group '{args.name}' does not exist")
            return
        print(f"group:    {info['name']}")
        print(f"roomId:   {info.get('roomId') or '(legacy, minted on first message)'}")
        print(f"members:  {', '.join(info['members']) or '—'}")
        return

    if action == "add":
        res = add_members(args.name, _split(args.bots))
        print(f"added {len(res['added'])} to '{args.name}'")
        return

    if action == "remove":
        res = remove_members(args.name, _split(args.bots))
        print(f"removed {len(res['removed'])} from '{args.name}'")
        return

    if action == "rename":
        res = rename_group(args.old, args.new)
        print(f"renamed '{res['old']}' → '{res['new']}'")
        return

    if action == "disband":
        disband_group(args.name)
        print(f"disbanded '{args.name}'")
        return

    raise SystemExit(f"unknown group action: {action}")

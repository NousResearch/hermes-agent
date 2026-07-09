"""Durable AI employee registry helpers for Hermes Desktop.

An AI employee is a human-facing wrapper around a Hermes profile/Agent.  The
stable system handle stays the profile id (lowercase ASCII slug); Chinese names
and role text live in ``agents/registry.json``, ``profile.yaml``, and the
profile's ``SOUL.md`` metadata block.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List

import yaml

from hermes_cli import profiles as profiles_mod

_log = logging.getLogger(__name__)

AGENT_PROFILE_META_START = "<!-- HERMES_AGENT_PROFILE_META_START -->"
AGENT_PROFILE_META_END = "<!-- HERMES_AGENT_PROFILE_META_END -->"
AGENT_PROFILE_META_RE = re.compile(
    rf"\n?{re.escape(AGENT_PROFILE_META_START)}.*?{re.escape(AGENT_PROFILE_META_END)}\n?",
    re.S,
)

METADATA_KEYS = ("display_name_zh", "role_zh", "mission_zh", "category", "emoji")


def registry_path() -> Path:
    return profiles_mod._get_default_hermes_home() / "agents" / "registry.json"


def read_registry() -> Dict[str, Any]:
    path = registry_path()
    if not path.is_file():
        return {"schema_version": 1, "agents": []}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _log.exception("Could not read AI employee registry at %s", path)
        return {"schema_version": 1, "agents": []}

    if not isinstance(data, dict):
        return {"schema_version": 1, "agents": []}
    if not isinstance(data.get("agents"), list):
        data["agents"] = []
    return data


def write_registry(registry: Dict[str, Any]) -> None:
    path = registry_path()
    agents = [agent for agent in registry.get("agents", []) if isinstance(agent, dict)]
    registry["agents"] = agents
    registry["count"] = len(agents)
    registry["updated_at"] = datetime.now(timezone.utc).isoformat()
    registry.setdefault("schema_version", 1)
    registry.setdefault("environment", "Hermes Desktop")
    registry.setdefault("hermes_home", str(profiles_mod._get_default_hermes_home()))
    registry.setdefault(
        "profile_id_policy",
        "[a-z0-9][a-z0-9_-]{0,63}; Chinese names are display metadata, not profile IDs.",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_registry_markdown(path.with_suffix(".md"), registry)


def write_registry_markdown(path: Path, registry: Dict[str, Any]) -> None:
    agents = sorted(
        [agent for agent in registry.get("agents", []) if isinstance(agent, dict)],
        key=lambda agent: int(agent.get("sort_order") or 10_000),
    )
    lines = [
        "# AI 员工 / Agent Registry",
        "",
        f"- Environment: {registry.get('environment') or 'Hermes Desktop'}",
        f"- HERMES_HOME: `{profiles_mod._get_default_hermes_home()}`",
        f"- Count: {len(agents)}",
        "- 说明：英文 `profile_id` 是系统调度 ID；中文名是 UI 显示名。",
        "",
        "| Profile ID | 中文显示名 | 岗位 | 分类 |",
        "|---|---|---|---|",
    ]
    for agent in agents:
        profile_id = str(agent.get("profile_id") or "")
        display_name = str(agent.get("display_name_zh") or profile_id)
        role = str(agent.get("role_zh") or "")
        category = str(agent.get("category") or "general")
        lines.append(f"| `{profile_id}` | {display_name} | {role} | `{category}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_profile_yaml(profile_dir: Path) -> Dict[str, Any]:
    path = profile_dir / "profile.yaml"
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_profile_yaml(profile_dir: Path, updates: Dict[str, Any]) -> None:
    data = read_profile_yaml(profile_dir)
    data.update(updates)
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "profile.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def parse_soul_metadata(soul_path: Path) -> Dict[str, str]:
    if not soul_path.is_file():
        return {}
    try:
        text = soul_path.read_text(encoding="utf-8")
    except OSError:
        return {}

    match = AGENT_PROFILE_META_RE.search(text)
    if not match:
        return {}

    meta: Dict[str, str] = {}
    for line in match.group(0).splitlines():
        stripped = line.strip()
        if stripped.startswith("- Profile ID:"):
            meta["profile_id"] = stripped.split(":", 1)[1].strip().strip("`")
        elif stripped.startswith("- 中文显示名:"):
            meta["display_name_zh"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("- 中文岗位:"):
            meta["role_zh"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("- Category:"):
            meta["category"] = stripped.split(":", 1)[1].strip().strip("`")
    return meta


def render_soul_metadata(agent: Dict[str, Any]) -> str:
    profile_id = str(agent.get("profile_id") or "")
    display_name = str(agent.get("display_name_zh") or profile_id)
    role = str(agent.get("role_zh") or "")
    category = str(agent.get("category") or "general")
    return (
        f"{AGENT_PROFILE_META_START}\n"
        "## Agent Profile Metadata\n\n"
        f"- Profile ID: `{profile_id}`\n"
        f"- 中文显示名: {display_name}\n"
        f"- 中文岗位: {role}\n"
        f"- Category: `{category}`\n"
        "- UI registry: `$HERMES_HOME/agents/registry.json`\n"
        f"{AGENT_PROFILE_META_END}\n\n"
    )


def write_soul_metadata(profile_dir: Path, agent: Dict[str, Any]) -> None:
    soul_path = profile_dir / "SOUL.md"
    try:
        body = soul_path.read_text(encoding="utf-8") if soul_path.is_file() else ""
    except OSError:
        body = ""
    body = AGENT_PROFILE_META_RE.sub("", body).lstrip("\n")
    soul_path.write_text(render_soul_metadata(agent) + body, encoding="utf-8")


def profile_to_dict(info: Any) -> Dict[str, Any]:
    def attr(name: str, default: Any = None) -> Any:
        try:
            return getattr(info, name)
        except Exception:
            return default

    return {
        "name": attr("name", ""),
        "path": str(attr("path", "")),
        "is_default": bool(attr("is_default", False)),
        "model": attr("model"),
        "provider": attr("provider"),
        "skill_count": int(attr("skill_count", 0) or 0),
        "gateway_running": bool(attr("gateway_running", False)),
        "description": attr("description", "") or "",
    }


def fallback_display_name(profile_id: str) -> str:
    return " ".join(part.capitalize() for part in profile_id.replace("_", "-").split("-") if part) or profile_id


def agent_employee_info(entry: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    profile_id = str(entry.get("profile_id") or profile.get("name") or "")
    profile_dir = Path(str(profile.get("path") or profiles_mod.get_profile_dir(profile_id)))
    profile_meta = read_profile_yaml(profile_dir)
    soul_meta = parse_soul_metadata(profile_dir / "SOUL.md")

    def pick(key: str, default: str = "") -> str:
        value = entry.get(key)
        if value is None or value == "":
            value = profile_meta.get(key)
        if value is None or value == "":
            value = soul_meta.get(key)
        if value is None or value == "":
            value = default
        return str(value)

    display_name = pick("display_name_zh", fallback_display_name(profile_id))
    mission = pick("mission_zh")
    description = str(profile.get("description") or profile_meta.get("description") or "")
    if not description and mission:
        description = f"{display_name}：{mission}"

    return {
        "profile_id": profile_id,
        "display_name_zh": display_name,
        "role_zh": pick("role_zh"),
        "mission_zh": mission,
        "category": pick("category", "general"),
        "emoji": pick("emoji", "🤖"),
        "sort_order": int(entry.get("sort_order") or profile_meta.get("sort_order") or 10_000),
        "profile_path": str(profile_dir),
        "soul_path": str(profile_dir / "SOUL.md"),
        "model": profile.get("model"),
        "provider": profile.get("provider"),
        "skill_count": int(profile.get("skill_count", 0) or 0),
        "gateway_running": bool(profile.get("gateway_running", False)),
        "description": description,
        "system_name_locked": True,
    }


def list_ai_employees() -> List[Dict[str, Any]]:
    profiles = [profile_to_dict(profile) for profile in profiles_mod.list_profiles()]
    profiles_by_name = {str(profile.get("name")): profile for profile in profiles}
    registry = read_registry()
    registry_entries = [agent for agent in registry.get("agents", []) if isinstance(agent, dict)]
    entries_by_profile = {
        str(agent.get("profile_id")): dict(agent)
        for agent in registry_entries
        if agent.get("profile_id")
    }

    if not entries_by_profile:
        entries_by_profile = {name: {"profile_id": name, "sort_order": idx} for idx, name in enumerate(profiles_by_name)}
    else:
        next_order = max((int(agent.get("sort_order") or 0) for agent in entries_by_profile.values()), default=0) + 10
        for name in profiles_by_name:
            if name not in entries_by_profile:
                entries_by_profile[name] = {"profile_id": name, "sort_order": next_order}
                next_order += 10

    employees: List[Dict[str, Any]] = []
    for profile_id, entry in entries_by_profile.items():
        profile = profiles_by_name.get(profile_id)
        if not profile:
            continue
        employees.append(agent_employee_info(entry, profile))
    return sorted(employees, key=lambda agent: (agent["sort_order"], agent["profile_id"]))


def normalize_metadata_updates(raw_updates: Dict[str, Any]) -> Dict[str, str]:
    updates: Dict[str, str] = {}
    for key in METADATA_KEYS:
        if key not in raw_updates or raw_updates[key] is None:
            continue
        text = str(raw_updates[key]).strip()
        if key in {"display_name_zh", "role_zh", "mission_zh"} and not text:
            raise ValueError(f"{key} cannot be empty")
        updates[key] = text
    return updates


def update_ai_employee_metadata(profile_id: str, raw_updates: Dict[str, Any]) -> Dict[str, Any]:
    profiles_mod.validate_profile_name(profile_id)
    if not profiles_mod.profile_exists(profile_id):
        raise FileNotFoundError(f"Profile '{profile_id}' does not exist.")

    updates = normalize_metadata_updates(raw_updates)
    profile_dir = profiles_mod.get_profile_dir(profile_id)
    registry = read_registry()
    entries = [agent for agent in registry.get("agents", []) if isinstance(agent, dict)]
    registry["agents"] = entries

    entry = next((agent for agent in entries if agent.get("profile_id") == profile_id), None)
    if entry is None:
        next_order = max((int(agent.get("sort_order") or 0) for agent in entries), default=0) + 10
        entry = {"profile_id": profile_id, "sort_order": next_order}
        entries.append(entry)

    entry.update(updates)
    entry["profile_id"] = profile_id
    entry["profile_path"] = str(profile_dir)
    entry["soul_path"] = str(profile_dir / "SOUL.md")
    entry["active"] = True
    entry["system_name_locked"] = True
    entry["rename_policy"] = (
        "profile_id must remain lowercase ASCII slug; Desktop should display "
        "display_name_zh as the human-facing name."
    )

    display_name = str(entry.get("display_name_zh") or fallback_display_name(profile_id))
    mission = str(entry.get("mission_zh") or "")
    profile_updates = {key: entry[key] for key in METADATA_KEYS if key in entry}
    profile_updates.update(
        {
            "description": f"{display_name}：{mission}" if mission else display_name,
            "description_auto": False,
            "sort_order": entry.get("sort_order"),
        }
    )
    write_profile_yaml(profile_dir, profile_updates)
    write_soul_metadata(profile_dir, entry)
    write_registry(registry)

    profile = next((profile_to_dict(item) for item in profiles_mod.list_profiles() if getattr(item, "name", None) == profile_id), None)
    if profile is None:
        raise FileNotFoundError(f"Profile '{profile_id}' does not exist.")
    return agent_employee_info(entry, profile)

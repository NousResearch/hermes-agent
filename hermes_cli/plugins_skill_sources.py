"""RAM-only plugin skill catalogs and authorized native slash payloads."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict

from hermes_cli.plugins_ledger import PluginRegistration
from hermes_cli.plugins_loader import _serialized_replacement

logger = logging.getLogger("hermes_cli.plugins")

@_serialized_replacement
def register_skill_source(
    self, name: str, *, list_skills: Callable, load_skill: Callable,
) -> PluginRegistration:
    """Register a scoped RAM-only catalog and text loader.

    Callbacks run in the caller's native profile/secret scope, never during
    registration. Catalog rows contain canonical name, description, opaque
    URI and optional frontmatter. Content is not persisted or preprocessed.
    """
    if not name or not callable(list_skills) or not callable(load_skill):
        raise ValueError("Skill source requires a name and callable catalog/loader")
    key = f"{self.plugin_id}:{name}"
    entry = {"list_skills": list_skills, "load_skill": load_skill,
             "plugin": self.manifest.name, "plugin_id": self.plugin_id}
    return self._register_entry("skill_source", key, self._manager._skill_sources,
                                entry, "Plugin %s registered skill source: %s", key)

def list_plugin_skill_metadata(self) -> list[dict[str, Any]]:
    """Return progressive-disclosure metadata for registered plugin skills."""
    return [
        {
            "name": qualified, "description": str(entry.get("description", "")),
            "category": "plugin", "frontmatter": dict(entry.get("frontmatter", {})),
        } for qualified, entry in sorted(self._plugin_skills.items())
    ]


def list_skill_source_commands(self) -> Dict[str, Dict[str, Any]]:
    """Fresh metadata only; no cross-profile or content cache."""
    from hermes_constants import get_hermes_home
    from agent.skill_commands import slugify_skill_name, skill_command_collision_note
    from agent.skill_utils import get_disabled_skill_names
    from tools.skills_tool import skill_matches_platform, skill_matches_apps, skill_matches_environment
    if Path(get_hermes_home()).resolve() != Path(self.home_path).resolve():
        return {}
    commands = {}
    ambiguous = set()
    disabled = get_disabled_skill_names()
    from hermes_cli.plugins_discovery import _get_disabled_plugins
    disabled_plugins = _get_disabled_plugins()
    for key, source in tuple(self._skill_sources.items()):
        if {source["plugin"], source["plugin_id"]} & disabled_plugins:
            continue
        try:
            rows = source["list_skills"]()
            if not isinstance(rows, list) or len(rows) > 1000:
                raise ValueError("Invalid skill source catalog")
            for row in rows:
                if not isinstance(row, dict):
                    continue
                name, uri = row.get("name"), row.get("uri")
                if (not isinstance(name, str) or not name or len(name) > 128
                        or slugify_skill_name(name) != name or name in disabled
                        or skill_command_collision_note(name) or not isinstance(uri, str) or not uri):
                    continue
                fm = row.get("frontmatter", {})
                if not isinstance(fm, dict) or not all((skill_matches_platform(fm),
                        skill_matches_apps(fm), skill_matches_environment(fm))):
                    continue
                command = f"/{name}"
                if command in commands or command in ambiguous:
                    commands.pop(command, None)
                    ambiguous.add(command)
                    logger.warning("Ambiguous remote skill command omitted: %s", command)
                    continue
                commands[command] = {"name": name,
                    "description": str(row.get("description", ""))[:500],
                    "source_id": key, "uri": uri}
        except Exception:  # health: allow BLE001 -- sanitize arbitrary plugin errors; do not log remote secrets
            logger.warning("Remote skill catalog unavailable for source %s", key)
    return commands

def load_skill_source_payload(self, info: Dict[str, Any]) -> Dict[str, Any]:
    """Reauthorize through a fresh catalog before fetching text in caller scope."""
    current = self.list_skill_source_commands().get(f"/{info.get('name', '')}")
    if current != info:
        return {"name": info.get("name", ""), "content": "[Remote skill unavailable: authorization changed.]", "remote": True}
    source = self._skill_sources.get(info.get("source_id"))
    try:
        payload = source["load_skill"](info["uri"])
        if (not isinstance(payload, dict) or payload.get("name") != info["name"]
                or not isinstance(payload.get("content"), str)
                or not payload["content"] or len(payload["content"]) > 256000):
            raise ValueError("Invalid remote skill text")
        latest = self.list_skill_source_commands().get(f"/{info['name']}")
        if latest != info or self._skill_sources.get(info['source_id']) is not source:
            return {"name": info['name'], "content": "[Remote skill unavailable: authorization changed.]", "remote": True}
        return {"name": info["name"], "content": payload["content"], "remote": True}
    except Exception:  # health: allow BLE001 -- sanitize arbitrary plugin errors; do not log remote secrets
        return {"name": info["name"], "content": "[Remote skill unavailable: check source permissions and connection.]", "remote": True}

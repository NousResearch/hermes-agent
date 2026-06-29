from __future__ import annotations

import json

from ..core.agent_config import load_presets_file, save_presets_file
from ..core.common import iso_now, print_json


def cmd_agent_config(args: list[str]) -> int:
    if not args:
        print_json({"ok": False, "error": "missing_subcommand"})
        return 1
    action = args[0]
    params = _flag_map(args[1:])
    file_path = params.get("file", "")
    name = params.get("name", "")
    config_json = params.get("config-json", "")
    if action == "list":
        if not file_path:
            print_json({"ok": False, "error": "missing_file"})
            return 1
        data = load_presets_file(file_path)
        presets = [{"name": preset["name"], "createdAt": preset["createdAt"]} for preset in data.get("presets", [])]
        print_json({"ok": True, "presets": presets, "count": len(presets)})
        return 0
    if action == "save":
        if not file_path or not name.strip() or not config_json.strip():
            print_json({"ok": False, "error": "missing_args"})
            return 1
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError:
            print_json({"ok": False, "error": "invalid_config_json"})
            return 1
        data = load_presets_file(file_path)
        action_name = "created"
        for preset in data["presets"]:
            if preset["name"].lower() == name.lower():
                preset["config"] = config
                preset["createdAt"] = iso_now()
                action_name = "updated"
                break
        else:
            data["presets"].append({"name": name, "createdAt": iso_now(), "config": config})
        save_presets_file(file_path, data)
        print_json({"ok": True, "name": name, "action": action_name})
        return 0
    if action == "load":
        if not file_path or not name.strip():
            print_json({"ok": False, "error": "missing_args"})
            return 1
        for preset in load_presets_file(file_path)["presets"]:
            if preset["name"].lower() == name.lower():
                print_json({"ok": True, "name": preset["name"], "config": preset["config"]})
                return 0
        print_json({"ok": False, "error": "preset_not_found", "name": name})
        return 1
    if action == "delete":
        if not file_path or not name.strip():
            print_json({"ok": False, "error": "missing_args"})
            return 1
        data = load_presets_file(file_path)
        filtered = [preset for preset in data["presets"] if preset["name"].lower() != name.lower()]
        if len(filtered) == len(data["presets"]):
            print_json({"ok": False, "error": "preset_not_found", "name": name})
            return 1
        data["presets"] = filtered
        save_presets_file(file_path, data)
        print_json({"ok": True, "name": name, "action": "deleted"})
        return 0
    print_json({"ok": False, "error": "unknown_subcommand", "subcommand": action})
    return 1


def _flag_map(args: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    index = 0
    while index < len(args):
        if args[index].startswith("--") and index + 1 < len(args):
            output[args[index][2:]] = args[index + 1]
            index += 2
            continue
        index += 1
    return output

from __future__ import annotations

from pathlib import Path
from typing import Any

FORBIDDEN_TRUE_PERMISSIONS = {
    "fleet_admin",
    "create_agents",
    "inspect_other_agents",
    "repair_other_agents",
    "gcp_admin",
    "tailscale_admin",
    "kidu_global_admin",
    "infisical_project_admin",
    "ssh_other_agents",
    "zeus_memory_access",
    "control_plane_access",
}


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none"}:
        return None
    return value.strip('"').strip("'")


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Small YAML subset parser for agent manifests.

    Supports top-level scalars, nested maps, and simple list items. This avoids
    requiring PyYAML during bootstrap while keeping manifests human-readable.
    """
    result: dict[str, Any] = {}
    current_key: str | None = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                result[key] = {}
                current_key = key
            else:
                result[key] = parse_scalar(value)
                current_key = None
            continue
        if current_key is None:
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            if not isinstance(result[current_key], list):
                result[current_key] = []
            result[current_key].append(parse_scalar(stripped[2:]))
        elif ":" in stripped:
            if not isinstance(result[current_key], dict):
                result[current_key] = {}
            key, value = stripped.split(":", 1)
            result[current_key][key.strip()] = parse_scalar(value)
    return result


def forbidden_permission_violations(manifest: dict[str, Any]) -> list[str]:
    permissions = manifest.get("permissions") or {}
    return sorted(k for k in FORBIDDEN_TRUE_PERMISSIONS if permissions.get(k) is True)

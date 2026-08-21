"""Read-only profile composition diagnostics for ``hermes doctor``.

This module intentionally reports only presence and non-secret configuration
metadata. It does not load dotenv files, inspect sessions, or mutate state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _profile_record(info: Any) -> dict[str, Any]:
    path = Path(info.path)
    config_path = path / "config.yaml"
    memory_dir = path / "memories"
    plugin_dir = path / "plugins"
    gateway_artifacts = (
        path / "gateway.pid",
        path / "gateway_state.json",
    )
    config: dict[str, Any] = {}
    if config_path.is_file():
        try:
            from hermes_cli.config import read_user_config_raw

            raw = read_user_config_raw(config_path)
            if isinstance(raw, dict):
                config = raw
        except Exception:
            pass
    memory_configured = isinstance(config.get("memory"), dict) and bool(
        config.get("memory")
    )
    plugins_configured = bool(config.get("plugins"))
    gateway_configured = bool(config.get("gateway"))
    return {
        "name": str(info.name),
        "status": "configured" if config_path.is_file() else "missing-config",
        "config_present": config_path.is_file(),
        "model": info.model,
        "provider": info.provider,
        "memory_present": memory_dir.exists() or memory_configured,
        "plugins_present": plugin_dir.exists() or plugins_configured,
        "gateway_present": any(item.exists() for item in gateway_artifacts)
        or gateway_configured,
        "gateway_running": bool(info.gateway_running),
    }


def build_profile_doctor_report(
    *, profile: str | None = None, all_profiles: bool = False
) -> dict[str, Any]:
    """Build a deterministic, JSON-serializable profile composition report."""
    from hermes_cli.profiles import list_profiles

    if profile is not None and all_profiles:
        raise ValueError("--profile and --all-profiles cannot be used together")

    profiles = sorted(
        list_profiles(), key=lambda item: (not item.is_default, item.name)
    )
    if profile is not None:
        selected = [item for item in profiles if item.name == profile]
        if not selected:
            raise ValueError(f"Profile '{profile}' does not exist")
    elif all_profiles:
        selected = profiles
    else:
        selected = [item for item in profiles if item.is_default][:1]
        if not selected and profiles:
            selected = profiles[:1]

    return {"profiles": [_profile_record(item) for item in selected]}


def render_profile_doctor_report(
    *, profile: str | None = None, all_profiles: bool = False, as_json: bool = False
) -> str:
    """Render the report for CLI output."""
    report = build_profile_doctor_report(profile=profile, all_profiles=all_profiles)
    if as_json:
        return json.dumps(report, sort_keys=True, separators=(",", ":"))
    lines = ["Profile composition:"]
    for item in report["profiles"]:
        lines.append(
            f"- {item['name']}: {item['status']} "
            f"(model={'present' if item['model'] else 'missing'}, "
            f"memory={'present' if item['memory_present'] else 'missing'}, "
            f"plugins={'present' if item['plugins_present'] else 'missing'}, "
            f"gateway={'running' if item['gateway_running'] else 'stopped'})"
        )
    return "\n".join(lines)

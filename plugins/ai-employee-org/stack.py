"""Operator stack and plugin enablement for AI employee org."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import yaml

from hermes_constants import get_hermes_home

from .core import PLUGIN_NAME, plugin_dir, stack_file


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def enable_plugin(*, dry_run: bool = False) -> dict[str, Any]:
    from hermes_cli.plugins_cmd import _get_enabled_set, _resolve_plugin_key, _save_enabled_set

    key = _resolve_plugin_key(PLUGIN_NAME)
    if key is None:
        return {"ok": False, "error": f"Plugin {PLUGIN_NAME} not discovered"}
    enabled = _get_enabled_set()
    if dry_run:
        return {"ok": True, "action": "would_enable" if key not in enabled else "already_enabled", "key": key}
    enabled.add(key)
    _save_enabled_set(enabled)
    return {"ok": True, "action": "enabled", "key": key}


def apply_ai_employee_stack(*, dry_run: bool = False) -> dict[str, Any]:
    """Merge bundled ai-employee-stack.yaml into ~/.hermes/config.yaml."""
    config_path = get_hermes_home() / "config.yaml"
    overlay_path = stack_file()
    if not overlay_path.is_file():
        return {"ok": False, "error": f"Missing stack file: {overlay_path}"}

    current = _load_yaml(config_path)
    overlay = _load_yaml(overlay_path)
    merged = _deep_merge(current, overlay)

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "config_path": str(config_path),
            "overlay_keys": sorted(overlay.keys()),
        }

    _save_yaml(config_path, merged)
    return {
        "ok": True,
        "config_path": str(config_path),
        "overlay_keys": sorted(overlay.keys()),
    }


def apply_stack_via_script(*, dry_run: bool = False) -> dict[str, Any]:
    """Fallback: invoke repo apply_operator_stack.py when yaml merge is insufficient."""
    repo_root = plugin_dir().parents[1]
    script = repo_root / "scripts" / "apply_operator_stack.py"
    if not script.is_file():
        return apply_ai_employee_stack(dry_run=dry_run)
    import subprocess

    cmd = [
        sys.executable,
        str(script),
        "--stack-file",
        str(stack_file()),
    ]
    if dry_run:
        cmd.append("--dry-run")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
    )
    return {
        "ok": proc.returncode == 0,
        "stdout": proc.stdout.strip()[:500],
        "stderr": proc.stderr.strip()[:500],
    }

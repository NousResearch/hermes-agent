#!/usr/bin/env python3
"""Check local Signal Room video production tool availability."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any


COMMANDS = {
    "blender": ("blender",),
    "moho": ("moho", "Moho"),
    "cavalry": ("cavalry", "Cavalry", "Cavalry.exe"),
    "hyperframes": ("npx",),
    "ffmpeg": ("ffmpeg",),
}
MOHO_ENV_VAR = "SIGNAL_ROOM_MOHO_EXE"
CAVALRY_ENV_VAR = "SIGNAL_ROOM_CAVALRY_EXE"
KNOWN_WINDOWS_MOHO_PATHS = (
    r"C:\Program Files\Moho 14\Moho.exe",
    "/mnt/c/Program Files/Moho 14/Moho.exe",
)
KNOWN_WINDOWS_CAVALRY_PATHS = (
    r"C:\Program Files\Cavalry\Cavalry.exe",
    "/mnt/c/Program Files/Cavalry/Cavalry.exe",
)


def _find_first(commands: tuple[str, ...], resolver: Callable[[str], str | None]) -> tuple[bool, str | None]:
    for command in commands:
        path = resolver(command)
        if path:
            return True, path
    return False, None


def _find_configured_app(env_var: str, known_paths: tuple[str, ...]) -> tuple[bool, str | None, str | None]:
    configured = os.getenv(env_var)
    if configured:
        return True, configured, env_var

    for path in known_paths:
        if Path(path).exists():
            return True, path, "known_windows_path"
    return False, None, None


def evaluate_environment(resolver: Callable[[str], str | None] = shutil.which) -> dict[str, Any]:
    tools: dict[str, Any] = {}
    for name, commands in COMMANDS.items():
        available, path = _find_first(commands, resolver)
        tools[name] = {
            "available": available,
            "path": path,
            "commands_checked": list(commands),
        }
    configured_moho_available, configured_moho_path, configured_moho_source = _find_configured_app(
        MOHO_ENV_VAR, KNOWN_WINDOWS_MOHO_PATHS
    )
    if configured_moho_available and not tools["moho"]["available"]:
        tools["moho"].update(
            {
                "available": True,
                "path": configured_moho_path,
                "source": configured_moho_source,
                "automation_mode": "windows_scheduled_task_bridge",
            }
        )
    configured_cavalry_available, configured_cavalry_path, configured_cavalry_source = _find_configured_app(
        CAVALRY_ENV_VAR, KNOWN_WINDOWS_CAVALRY_PATHS
    )
    if configured_cavalry_available and not tools["cavalry"]["available"]:
        tools["cavalry"].update(
            {
                "available": True,
                "path": configured_cavalry_path,
                "source": configured_cavalry_source,
                "automation_mode": "windows_interactive_app",
                "role": "optional_motion_graphics",
            }
        )

    blockers: list[str] = []
    has_pose_renderer = tools["blender"]["available"] or tools["moho"]["available"]
    if not has_pose_renderer:
        blockers.append("Blender or Moho is required for local character pose rendering")
    if not tools["hyperframes"]["available"]:
        blockers.append("npx is required for HyperFrames preview/render commands")
    if not tools["ffmpeg"]["available"]:
        blockers.append("ffmpeg is required for final video rendering")

    if tools["blender"]["available"]:
        render_mode = "local_blender"
    elif tools["moho"]["available"]:
        render_mode = "local_moho"
    else:
        render_mode = "external_pose_export_required"

    return {
        "passed": not blockers,
        "render_mode": render_mode,
        "blockers": blockers,
        "tools": tools,
        "next_action": _next_action(render_mode, blockers),
    }


def _next_action(render_mode: str, blockers: list[str]) -> str:
    if not blockers:
        return "render candidate poses locally, then run rig/contact/retention gates"
    if render_mode == "external_pose_export_required":
        return "install Blender locally or export pose PNGs from a Blender/Moho-capable machine"
    return "install missing HyperFrames/FFmpeg support before preview or final render"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write environment gate JSON to this path")
    args = parser.parse_args()

    result = evaluate_environment()
    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

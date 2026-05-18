"""Handlers for the build-macos-apps plugin."""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tools.registry import tool_error, tool_result

_DISCOVERY_MAX_DEPTH = 3
_XCODEBUILD_PATH = "xcodebuild"
_SIGNING_DISABLED_FLAGS = [
    "CODE_SIGNING_ALLOWED=NO",
    "CODE_SIGNING_REQUIRED=NO",
    "CODE_SIGN_IDENTITY=",
]


def check_macos_dev_requirements() -> bool:
    """Return True when Hermes can expose the macOS build toolset."""
    return sys_platform_is_darwin() and bool(shutil.which(_XCODEBUILD_PATH))


def sys_platform_is_darwin() -> bool:
    return platform.system().lower() == "darwin"


def _normalize_path(raw: str | os.PathLike[str]) -> Path:
    if raw is None:
        raise ValueError("path is required")
    path = Path(raw).expanduser()
    try:
        resolved = path.resolve()
    except FileNotFoundError:
        resolved = path.absolute()
    return resolved


def _coerce_timeout(raw: Any, default: int = 1800) -> int:
    try:
        timeout = int(raw)
    except Exception:
        timeout = default
    return max(30, min(7200, timeout))


def _iter_candidates(root: Path, suffix: str) -> Iterable[Path]:
    if root.is_file():
        if root.suffix == suffix:
            yield root
        return

    seen: set[Path] = set()
    for current_root, dirs, files in os.walk(root):
        current = Path(current_root)
        rel_parts = current.relative_to(root).parts if current != root else ()
        if len(rel_parts) > _DISCOVERY_MAX_DEPTH:
            dirs[:] = []
            continue

        if suffix == ".xcodeproj":
            names = [name for name in dirs if name.endswith(suffix)]
        else:
            names = [name for name in dirs if name.endswith(suffix)]

        for name in sorted(names):
            candidate = current / name
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def _inspect_project(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    root = path if path.is_dir() else path.parent
    workspaces = list(_iter_candidates(path, ".xcworkspace"))
    projects = list(_iter_candidates(path, ".xcodeproj"))
    package_swift = sorted(
        p for p in root.rglob("Package.swift")
        if len(p.relative_to(root).parts) <= _DISCOVERY_MAX_DEPTH + 1
    ) if root.exists() else []

    primary: Path | None = None
    primary_type = "none"
    if workspaces:
        primary = workspaces[0]
        primary_type = "workspace"
    elif projects:
        primary = projects[0]
        primary_type = "project"

    return {
        "input_path": str(path),
        "project_root": str(root),
        "exists": True,
        "xcode_workspaces": [str(p) for p in workspaces],
        "xcode_projects": [str(p) for p in projects],
        "swift_packages": [str(p) for p in package_swift],
        "has_package_swift": bool(package_swift),
        "recommended_container": str(primary) if primary else None,
        "recommended_container_type": primary_type,
        "supports_xcodebuild_flow": bool(primary),
        "notes": _build_inspection_notes(workspaces, projects, package_swift),
    }


def _build_inspection_notes(
    workspaces: List[Path], projects: List[Path], packages: List[Path]
) -> List[str]:
    notes: List[str] = []
    if workspaces:
        notes.append("Workspace detected; prefer the workspace for scheme listing and builds.")
    elif projects:
        notes.append("Project detected; scheme listing and builds can target the .xcodeproj directly.")
    else:
        notes.append("No .xcworkspace or .xcodeproj was found within the inspection depth.")

    if packages and not (workspaces or projects):
        notes.append("Swift Package manifest detected, but Phase 1 only exposes Xcode project/workspace flows.")
    elif packages:
        notes.append("Swift Package manifest detected alongside Xcode build metadata.")
    return notes


def _select_container(base_path: Path, container_path: str | None) -> Dict[str, str]:
    explicit = None
    if container_path:
        candidate = Path(container_path).expanduser()
        if not candidate.is_absolute():
            base_dir = base_path if base_path.is_dir() else base_path.parent
            candidate = (base_dir / candidate).resolve()
        explicit = _normalize_path(candidate)
    if explicit:
        if not explicit.exists():
            raise FileNotFoundError(f"Container not found: {explicit}")
        if explicit.suffix not in {".xcworkspace", ".xcodeproj"}:
            raise ValueError("container_path must point to a .xcworkspace or .xcodeproj")
        return {"type": "workspace" if explicit.suffix == ".xcworkspace" else "project", "path": str(explicit)}

    inspection = _inspect_project(base_path)
    recommended = inspection.get("recommended_container")
    if not recommended:
        raise ValueError(
            "No .xcworkspace or .xcodeproj found. Run macos_inspect_project first and point Hermes at an Xcode container."
        )
    return {
        "type": inspection["recommended_container_type"],
        "path": str(recommended),
    }


def _run_xcodebuild(
    command: List[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def _shape_xcode_output(stdout: str, stderr: str, *, max_tail_lines: int = 40) -> Dict[str, Any]:
    combined = "\n".join(part for part in [stdout.strip(), stderr.strip()] if part).strip()
    if not combined:
        return {"tail": [], "highlights": [], "line_count": 0}

    lines = combined.splitlines()
    highlights: List[str] = []
    pattern = re.compile(r"(error:|warning:|\*\* BUILD (SUCCEEDED|FAILED) \*\*|Testing failed:)", re.IGNORECASE)
    for line in lines:
        if pattern.search(line):
            highlights.append(line)
        if len(highlights) >= 20:
            break

    tail = lines[-max_tail_lines:]
    return {
        "tail": tail,
        "highlights": highlights,
        "line_count": len(lines),
    }


def _json_loads_or_error(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"xcodebuild returned non-JSON output: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("xcodebuild returned an unexpected JSON payload")
    return data


def handle_macos_inspect_project(args: dict, **kw) -> str:
    try:
        path = _normalize_path(args.get("path"))
        inspection = _inspect_project(path)
        return tool_result({"success": True, **inspection})
    except Exception as exc:
        return tool_error(str(exc), success=False)


def handle_macos_list_schemes(args: dict, **kw) -> str:
    try:
        path = _normalize_path(args.get("path"))
        container = _select_container(path, args.get("container_path"))
        command = [_XCODEBUILD_PATH, "-list", "-json", f"-{container['type']}", container["path"]]
        completed = _run_xcodebuild(command, cwd=path if path.is_dir() else path.parent, timeout_seconds=120)
        if completed.returncode != 0:
            shaped = _shape_xcode_output(completed.stdout, completed.stderr)
            return tool_error(
                "xcodebuild -list failed",
                success=False,
                exit_code=completed.returncode,
                command=shlex.join(command),
                container=container,
                output=shaped,
            )

        payload = _json_loads_or_error(completed.stdout)
        return tool_result(
            {
                "success": True,
                "command": shlex.join(command),
                "container": container,
                "project": payload.get("project"),
            }
        )
    except subprocess.TimeoutExpired:
        return tool_error("xcodebuild -list timed out", success=False)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def handle_macos_build_project(args: dict, **kw) -> str:
    try:
        path = _normalize_path(args.get("path"))
        scheme = str(args.get("scheme") or "").strip()
        if not scheme:
            return tool_error("scheme is required", success=False)

        container = _select_container(path, args.get("container_path"))
        configuration = str(args.get("configuration") or "Debug").strip() or "Debug"
        destination = str(args.get("destination") or "generic/platform=macOS").strip() or "generic/platform=macOS"
        timeout_seconds = _coerce_timeout(args.get("timeout_seconds"), default=1800)

        command = [
            _XCODEBUILD_PATH,
            "build",
            f"-{container['type']}",
            container["path"],
            "-scheme",
            scheme,
            "-configuration",
            configuration,
            "-destination",
            destination,
        ]
        derived_data_path = args.get("derived_data_path")
        if derived_data_path:
            derived_path = Path(str(derived_data_path)).expanduser()
            if not derived_path.is_absolute():
                base_dir = path if path.is_dir() else path.parent
                derived_path = (base_dir / derived_path).resolve()
            command.extend(["-derivedDataPath", str(_normalize_path(derived_path))])
        command.extend(_SIGNING_DISABLED_FLAGS)

        started = time.time()
        completed = _run_xcodebuild(
            command,
            cwd=path if path.is_dir() else path.parent,
            timeout_seconds=timeout_seconds,
        )
        duration_seconds = round(time.time() - started, 2)
        shaped = _shape_xcode_output(completed.stdout, completed.stderr)
        success = completed.returncode == 0

        result = {
            "success": success,
            "command": shlex.join(command),
            "container": container,
            "scheme": scheme,
            "configuration": configuration,
            "destination": destination,
            "unsigned_build": True,
            "duration_seconds": duration_seconds,
            "exit_code": completed.returncode,
            "output": shaped,
        }
        if success:
            return tool_result(result)
        return tool_error("xcodebuild build failed", **result)
    except subprocess.TimeoutExpired as exc:
        return tool_error(
            "xcodebuild build timed out",
            success=False,
            timeout_seconds=exc.timeout,
        )
    except Exception as exc:
        return tool_error(str(exc), success=False)

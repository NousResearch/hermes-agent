"""Handlers for the build-macos-apps plugin."""

from __future__ import annotations

import json
import os
import platform
import plistlib
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


def _coerce_small_timeout(raw: Any, default: int, minimum: int = 0, maximum: int = 60) -> int:
    try:
        timeout = int(raw)
    except Exception:
        timeout = default
    return max(minimum, min(maximum, timeout))


def _coerce_string_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    item = str(raw).strip()
    return [item] if item else []


def _coerce_bool(raw: Any, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    cleaned = str(raw).strip().lower()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    return default


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


def _resolve_optional_path(base_path: Path, raw_path: Any) -> str | None:
    if not raw_path:
        return None
    candidate = Path(str(raw_path)).expanduser()
    if not candidate.is_absolute():
        base_dir = base_path if base_path.is_dir() else base_path.parent
        candidate = (base_dir / candidate).resolve()
    return str(_normalize_path(candidate))


def _normalize_app_name(raw: Any) -> str | None:
    if raw is None:
        return None
    cleaned = str(raw).strip()
    if not cleaned:
        return None
    return cleaned[:-4] if cleaned.lower().endswith(".app") else cleaned


def _find_app_bundles(
    path: Path,
    *,
    app_name: str | None = None,
    configuration: str = "Debug",
    derived_data_path: str | None = None,
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    root = path if path.is_dir() else path.parent
    normalized_app_name = _normalize_app_name(app_name)
    preferred_config = (configuration or "Debug").strip() or "Debug"

    search_roots: List[Path] = []
    if derived_data_path:
        search_roots.append(Path(derived_data_path))
    search_roots.extend(
        [
            root / "DerivedData",
            root / "build",
            root / "Build",
            root / "dist",
            root,
        ]
    )

    unique_roots: List[Path] = []
    seen_roots: set[Path] = set()
    for search_root in search_roots:
        resolved = search_root.resolve() if search_root.exists() else search_root
        if not search_root.exists() or resolved in seen_roots:
            continue
        seen_roots.add(resolved)
        unique_roots.append(search_root)

    matches: List[Dict[str, Any]] = []
    seen_apps: set[Path] = set()
    for search_root in unique_roots:
        for app_path in search_root.rglob("*.app"):
            if app_path in seen_apps:
                continue
            seen_apps.add(app_path)
            if normalized_app_name and app_path.stem != normalized_app_name:
                continue

            score = 0
            app_string = str(app_path)
            if preferred_config.lower() in app_string.lower():
                score += 20
            if "Build/Products" in app_string:
                score += 30
            if derived_data_path and Path(derived_data_path) in app_path.parents:
                score += 40
            rel_parts = app_path.relative_to(search_root).parts if app_path.is_relative_to(search_root) else ()
            score += max(0, 10 - len(rel_parts))

            matches.append(
                {
                    "path": str(app_path),
                    "name": app_path.stem,
                    "search_root": str(search_root),
                    "score": score,
                }
            )

    matches.sort(key=lambda item: (-item["score"], item["path"]))
    return {
        "success": True,
        "app_name_filter": normalized_app_name,
        "configuration": preferred_config,
        "derived_data_path": derived_data_path,
        "matches": matches,
        "recommended_app_bundle": matches[0]["path"] if matches else None,
    }


def _resolve_app_bundle(
    path: Path,
    *,
    app_bundle_path: Any = None,
    app_name: Any = None,
    configuration: str = "Debug",
    derived_data_path: str | None = None,
) -> Dict[str, Any]:
    explicit_bundle = _resolve_optional_path(path, app_bundle_path)
    if explicit_bundle:
        bundle_path = Path(explicit_bundle)
        if bundle_path.suffix != ".app":
            raise ValueError("app_bundle_path must point to a .app bundle")
        if not bundle_path.exists():
            raise FileNotFoundError(f"App bundle not found: {bundle_path}")
        return {
            "bundle_path": str(bundle_path),
            "bundle_name": bundle_path.stem,
            "discovered": False,
            "candidates": [],
        }

    discovered = _find_app_bundles(
        path,
        app_name=app_name,
        configuration=configuration,
        derived_data_path=derived_data_path,
    )
    recommended = discovered.get("recommended_app_bundle")
    if not recommended:
        raise ValueError("No .app bundle found. Build the app first or pass app_bundle_path explicitly.")

    bundle_path = Path(recommended)
    return {
        "bundle_path": str(bundle_path),
        "bundle_name": bundle_path.stem,
        "discovered": True,
        "candidates": discovered.get("matches", []),
    }


def _read_bundle_identifier(app_bundle_path: Path) -> str | None:
    info_plist = app_bundle_path / "Contents" / "Info.plist"
    if not info_plist.exists():
        return None
    try:
        with info_plist.open("rb") as fh:
            payload = plistlib.load(fh)
    except Exception:
        return None
    bundle_identifier = payload.get("CFBundleIdentifier")
    return str(bundle_identifier).strip() if bundle_identifier else None


def _pgrep_app(app_name: str) -> List[int]:
    completed = subprocess.run(
        ["pgrep", "-ix", app_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        return []
    pids: List[int] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _wait_for_app_state(app_name: str, *, should_be_running: bool, timeout_seconds: int) -> List[int]:
    deadline = time.time() + timeout_seconds
    while True:
        pids = _pgrep_app(app_name)
        if should_be_running and pids:
            return pids
        if not should_be_running and not pids:
            return []
        if time.time() >= deadline:
            return pids
        time.sleep(0.25)


def _run_system_command(
    command: List[str],
    *,
    cwd: Path | None = None,
    timeout_seconds: int = 300,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


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


def _parse_build_settings_output(raw: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    for line in raw.splitlines():
        stripped = line.rstrip()
        if not stripped:
            continue
        if stripped.startswith("Build settings for action ") and ":" in stripped:
            current = {"header": stripped, "settings": {}}
            sections.append(current)
            continue
        if current is None or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        current["settings"][key.strip()] = value.strip()
    return sections


def _extract_build_settings_focus(sections: List[Dict[str, Any]], target_name: str | None = None) -> Dict[str, Any]:
    if not sections:
        return {"section_count": 0, "selected_header": None, "settings": {}, "interesting": {}}

    selected = sections[0]
    if target_name:
        lowered = target_name.lower()
        for section in sections:
            if lowered in section["header"].lower():
                selected = section
                break

    settings = selected["settings"]
    interesting_keys = [
        "TARGET_NAME",
        "PRODUCT_NAME",
        "FULL_PRODUCT_NAME",
        "PRODUCT_BUNDLE_IDENTIFIER",
        "CONFIGURATION",
        "SDKROOT",
        "BUILT_PRODUCTS_DIR",
        "TARGET_BUILD_DIR",
        "WRAPPER_NAME",
        "EXECUTABLE_PATH",
        "INFOPLIST_FILE",
    ]
    interesting = {key: settings[key] for key in interesting_keys if key in settings}
    return {
        "section_count": len(sections),
        "selected_header": selected["header"],
        "settings": settings,
        "interesting": interesting,
    }


def _coerce_recent_window(raw: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _coerce_limit(raw: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _diagnostic_reports_dir() -> Path:
    return Path.home() / "Library" / "Logs" / "DiagnosticReports"


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
        derived_data_path = _resolve_optional_path(path, args.get("derived_data_path"))
        if derived_data_path:
            command.extend(["-derivedDataPath", derived_data_path])
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


def handle_macos_test_project(args: dict, **kw) -> str:
    try:
        path = _normalize_path(args.get("path"))
        scheme = str(args.get("scheme") or "").strip()
        if not scheme:
            return tool_error("scheme is required", success=False)

        container = _select_container(path, args.get("container_path"))
        configuration = str(args.get("configuration") or "Debug").strip() or "Debug"
        destination = str(args.get("destination") or "platform=macOS").strip() or "platform=macOS"
        test_plan = str(args.get("test_plan") or "").strip() or None
        only_testing = _coerce_string_list(args.get("only_testing"))
        skip_testing = _coerce_string_list(args.get("skip_testing"))
        timeout_seconds = _coerce_timeout(args.get("timeout_seconds"), default=1800)

        command = [
            _XCODEBUILD_PATH,
            "test",
            f"-{container['type']}",
            container["path"],
            "-scheme",
            scheme,
            "-configuration",
            configuration,
            "-destination",
            destination,
        ]
        if test_plan:
            command.extend(["-testPlan", test_plan])
        for identifier in only_testing:
            command.extend(["-only-testing", identifier])
        for identifier in skip_testing:
            command.extend(["-skip-testing", identifier])

        derived_data_path = _resolve_optional_path(path, args.get("derived_data_path"))
        if derived_data_path:
            command.extend(["-derivedDataPath", derived_data_path])

        result_bundle_path = _resolve_optional_path(path, args.get("result_bundle_path"))
        if result_bundle_path:
            command.extend(["-resultBundlePath", result_bundle_path])

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
            "test_plan": test_plan,
            "only_testing": only_testing,
            "skip_testing": skip_testing,
            "result_bundle_path": result_bundle_path,
            "signing_disabled": True,
            "duration_seconds": duration_seconds,
            "exit_code": completed.returncode,
            "output": shaped,
        }
        if success:
            return tool_result(result)
        return tool_error("xcodebuild test failed", **result)
    except subprocess.TimeoutExpired as exc:
        return tool_error(
            "xcodebuild test timed out",
            success=False,
            timeout_seconds=exc.timeout,
        )
    except Exception as exc:
        return tool_error(str(exc), success=False)


def handle_macos_find_app_bundle(args: dict, **kw) -> str:
    try:
        path = _normalize_path(args.get("path"))
        configuration = str(args.get("configuration") or "Debug").strip() or "Debug"
        derived_data_path = _resolve_optional_path(path, args.get("derived_data_path"))
        result = _find_app_bundles(
            path,
            app_name=args.get("app_name"),
            configuration=configuration,
            derived_data_path=derived_data_path,
        )
        return tool_result(result)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def handle_macos_run_app(args: dict, **kw) -> str:
    try:
        path = _normalize_path(args.get("path"))
        configuration = str(args.get("configuration") or "Debug").strip() or "Debug"
        derived_data_path = _resolve_optional_path(path, args.get("derived_data_path"))
        resolved = _resolve_app_bundle(
            path,
            app_bundle_path=args.get("app_bundle_path"),
            app_name=args.get("app_name"),
            configuration=configuration,
            derived_data_path=derived_data_path,
        )
        bundle_path = resolved["bundle_path"]
        bundle_name = resolved["bundle_name"]

        command = ["open"]
        if _coerce_bool(args.get("new_instance"), default=False):
            command.append("-n")
        if not _coerce_bool(args.get("activate"), default=True):
            command.append("-g")
        command.append(bundle_path)

        app_args = _coerce_string_list(args.get("args"))
        if app_args:
            command.append("--args")
            command.extend(app_args)

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        pids = _wait_for_app_state(
            bundle_name,
            should_be_running=True,
            timeout_seconds=_coerce_small_timeout(args.get("wait_running_seconds"), default=5),
        )
        success = completed.returncode == 0
        result = {
            "success": success,
            "command": shlex.join(command),
            "app_bundle_path": bundle_path,
            "app_name": bundle_name,
            "discovered_bundle": resolved["discovered"],
            "candidate_bundles": resolved["candidates"],
            "args": app_args,
            "is_running": bool(pids),
            "pids": pids,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "exit_code": completed.returncode,
        }
        if success:
            return tool_result(result)
        return tool_error("Failed to launch app with open", **result)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def handle_macos_stop_app(args: dict, **kw) -> str:
    try:
        path = _normalize_path(args.get("path"))
        configuration = str(args.get("configuration") or "Debug").strip() or "Debug"
        derived_data_path = _resolve_optional_path(path, args.get("derived_data_path"))
        resolved = _resolve_app_bundle(
            path,
            app_bundle_path=args.get("app_bundle_path"),
            app_name=args.get("app_name"),
            configuration=configuration,
            derived_data_path=derived_data_path,
        )
        bundle_path = Path(resolved["bundle_path"])
        bundle_name = resolved["bundle_name"]
        bundle_identifier = _read_bundle_identifier(bundle_path)
        force = _coerce_bool(args.get("force"), default=False)
        timeout_seconds = _coerce_small_timeout(args.get("timeout_seconds"), default=15)

        initial_pids = _pgrep_app(bundle_name)
        if not initial_pids:
            return tool_result(
                {
                    "success": True,
                    "app_bundle_path": str(bundle_path),
                    "app_name": bundle_name,
                    "bundle_identifier": bundle_identifier,
                    "was_running": False,
                    "stopped": True,
                    "force": force,
                    "pids": [],
                }
            )

        actions: List[str] = []
        apple_stdout = ""
        apple_stderr = ""
        if bundle_identifier:
            applescript = ["osascript", "-e", f'tell application id "{bundle_identifier}" to quit']
            apple_completed = subprocess.run(
                applescript,
                capture_output=True,
                text=True,
                check=False,
            )
            actions.append(shlex.join(applescript))
            apple_stdout = apple_completed.stdout.strip()
            apple_stderr = apple_completed.stderr.strip()

        remaining_pids = _wait_for_app_state(
            bundle_name,
            should_be_running=False,
            timeout_seconds=timeout_seconds,
        )

        kill_stdout = ""
        kill_stderr = ""
        if remaining_pids:
            kill_command = ["pkill", "-KILL" if force else "-TERM", "-ix", bundle_name]
            kill_completed = subprocess.run(
                kill_command,
                capture_output=True,
                text=True,
                check=False,
            )
            actions.append(shlex.join(kill_command))
            kill_stdout = kill_completed.stdout.strip()
            kill_stderr = kill_completed.stderr.strip()
            remaining_pids = _wait_for_app_state(
                bundle_name,
                should_be_running=False,
                timeout_seconds=2,
            )

        stopped = not remaining_pids
        result = {
            "success": stopped,
            "app_bundle_path": str(bundle_path),
            "app_name": bundle_name,
            "bundle_identifier": bundle_identifier,
            "was_running": True,
            "stopped": stopped,
            "force": force,
            "initial_pids": initial_pids,
            "remaining_pids": remaining_pids,
            "actions": actions,
            "stdout": "\n".join(part for part in [apple_stdout, kill_stdout] if part),
            "stderr": "\n".join(part for part in [apple_stderr, kill_stderr] if part),
        }
        if stopped:
            return tool_result(result)
        return tool_error("Failed to stop app cleanly", **result)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def handle_macos_read_recent_logs(args: dict, **kw) -> str:
    try:
        path = _normalize_path(args.get("path"))
        configuration = str(args.get("configuration") or "Debug").strip() or "Debug"
        derived_data_path = _resolve_optional_path(path, args.get("derived_data_path"))
        predicate = str(args.get("predicate") or "").strip() or None

        resolved = _resolve_app_bundle(
            path,
            app_bundle_path=args.get("app_bundle_path"),
            app_name=args.get("app_name"),
            configuration=configuration,
            derived_data_path=derived_data_path,
        )
        bundle_path = Path(resolved["bundle_path"])
        bundle_name = resolved["bundle_name"]
        bundle_identifier = _read_bundle_identifier(bundle_path)

        if predicate is None:
            if bundle_identifier:
                predicate = f'processImagePath ENDSWITH "{bundle_path.name}" OR senderImagePath ENDSWITH "{bundle_path.name}" OR eventMessage CONTAINS[c] "{bundle_identifier}"'
            else:
                predicate = f'process == "{bundle_name}"'

        since_minutes = _coerce_recent_window(args.get("since_minutes"), default=15, minimum=1, maximum=1440)
        limit = _coerce_limit(args.get("limit"), default=200, minimum=1, maximum=1000)
        command = [
            "log",
            "show",
            "--style",
            "compact",
            "--last",
            f"{since_minutes}m",
            "--predicate",
            predicate,
        ]
        completed = _run_system_command(command, timeout_seconds=300)
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        limited_lines = lines[-limit:]
        result = {
            "success": completed.returncode == 0,
            "command": shlex.join(command),
            "app_bundle_path": str(bundle_path),
            "app_name": bundle_name,
            "bundle_identifier": bundle_identifier,
            "predicate": predicate,
            "since_minutes": since_minutes,
            "line_count": len(lines),
            "lines": limited_lines,
            "truncated": len(lines) > len(limited_lines),
            "stderr": completed.stderr.strip(),
            "exit_code": completed.returncode,
        }
        if completed.returncode == 0:
            return tool_result(result)
        return tool_error("log show failed", **result)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def handle_macos_collect_crash_reports(args: dict, **kw) -> str:
    try:
        path = _normalize_path(args.get("path"))
        configuration = str(args.get("configuration") or "Debug").strip() or "Debug"
        derived_data_path = _resolve_optional_path(path, args.get("derived_data_path"))
        resolved = _resolve_app_bundle(
            path,
            app_bundle_path=args.get("app_bundle_path"),
            app_name=args.get("app_name"),
            configuration=configuration,
            derived_data_path=derived_data_path,
        )
        bundle_name = resolved["bundle_name"]
        since_hours = _coerce_recent_window(args.get("since_hours"), default=24, minimum=1, maximum=720)
        limit = _coerce_limit(args.get("limit"), default=10, minimum=1, maximum=100)
        cutoff = time.time() - since_hours * 3600

        reports_dir = _diagnostic_reports_dir()
        reports: List[Dict[str, Any]] = []
        if reports_dir.exists():
            for candidate in sorted(reports_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if candidate.suffix.lower() not in {".crash", ".ips", ".spin", ".hang"}:
                    continue
                if not candidate.name.startswith(bundle_name):
                    continue
                stat = candidate.stat()
                if stat.st_mtime < cutoff:
                    continue
                try:
                    preview_lines = candidate.read_text(errors="replace").splitlines()[:20]
                except Exception:
                    preview_lines = []
                reports.append(
                    {
                        "path": str(candidate),
                        "filename": candidate.name,
                        "modified_at_epoch": stat.st_mtime,
                        "size_bytes": stat.st_size,
                        "preview": preview_lines,
                    }
                )
                if len(reports) >= limit:
                    break

        return tool_result(
            {
                "success": True,
                "app_name": bundle_name,
                "reports_dir": str(reports_dir),
                "since_hours": since_hours,
                "report_count": len(reports),
                "reports": reports,
            }
        )
    except Exception as exc:
        return tool_error(str(exc), success=False)


def handle_macos_show_build_settings(args: dict, **kw) -> str:
    try:
        path = _normalize_path(args.get("path"))
        scheme = str(args.get("scheme") or "").strip()
        if not scheme:
            return tool_error("scheme is required", success=False)
        container = _select_container(path, args.get("container_path"))
        configuration = str(args.get("configuration") or "Debug").strip() or "Debug"
        destination = str(args.get("destination") or "").strip() or None
        target_name = str(args.get("target_name") or "").strip() or None
        timeout_seconds = _coerce_timeout(args.get("timeout_seconds"), default=300)

        command = [
            _XCODEBUILD_PATH,
            "-showBuildSettings",
            f"-{container['type']}",
            container["path"],
            "-scheme",
            scheme,
            "-configuration",
            configuration,
        ]
        if destination:
            command.extend(["-destination", destination])

        completed = _run_xcodebuild(
            command,
            cwd=path if path.is_dir() else path.parent,
            timeout_seconds=timeout_seconds,
        )
        if completed.returncode != 0:
            shaped = _shape_xcode_output(completed.stdout, completed.stderr)
            return tool_error(
                "xcodebuild -showBuildSettings failed",
                success=False,
                command=shlex.join(command),
                container=container,
                exit_code=completed.returncode,
                output=shaped,
            )

        sections = _parse_build_settings_output(completed.stdout)
        focused = _extract_build_settings_focus(sections, target_name=target_name)
        return tool_result(
            {
                "success": True,
                "command": shlex.join(command),
                "container": container,
                "scheme": scheme,
                "configuration": configuration,
                "target_name": target_name,
                "destination": destination,
                **focused,
            }
        )
    except subprocess.TimeoutExpired as exc:
        return tool_error("xcodebuild -showBuildSettings timed out", success=False, timeout_seconds=exc.timeout)
    except Exception as exc:
        return tool_error(str(exc), success=False)

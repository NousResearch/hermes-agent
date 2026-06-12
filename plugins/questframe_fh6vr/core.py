"""Core implementation for the QuestFrame Hermes plugin."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover - early import safety
    def get_hermes_home() -> Path:  # type: ignore[no-redef]
        return Path.home() / ".hermes"


PLUGIN_ID = "questframe-fh6vr"
PLUGIN_NAME = "questframe-fh6vr"
CONFIG_ALIASES = (PLUGIN_ID, "questframe_fh6vr", "questframe")
DEFAULT_TIMEOUT_SECONDS = 60

KNOWN_VPM_PACKAGES = {
    "com.vrchat.avatars": "VRChat SDK Avatars",
    "com.vrchat.worlds": "VRChat SDK Worlds",
    "com.vrchat.base": "VRChat SDK Base",
    "nadena.dev.modular-avatar": "Modular Avatar",
    "nadena.dev.ndmf": "NDMF",
    "jp.lilxyzw.liltoon": "lilToon",
    "com.poiyomi.toon": "Poiyomi Toon",
    "com.poiyomi.shader": "Poiyomi Shader",
    "com.vrcfury.vrcfury": "VRCFury",
    "com.anatawa12.avatar-optimizer": "Avatar Optimizer",
    "com.blackstartx.gesture-manager": "Gesture Manager",
    "com.vrchat.gesture-manager": "Gesture Manager",
}

STATUS_SCHEMA = {
    "name": "questframe_status",
    "description": "Show QuestFrame Hermes bridge readiness without mutating user files.",
    "parameters": {"type": "object", "properties": {}},
}

SETUP_SCHEMA = {
    "name": "questframe_setup",
    "description": "Save non-secret QuestFrame bridge paths to Hermes config.yaml.",
    "parameters": {
        "type": "object",
        "properties": {
            "launcher_exe": {
                "type": "string",
                "description": "Path to FH6VR.Launcher executable.",
            },
            "vcc_project_roots": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Unity/VCC project roots to scan.",
            },
            "unity_python": {
                "type": "string",
                "description": "Optional Python executable for future Unity bridge helpers.",
            },
        },
    },
}

PREFLIGHT_SCHEMA = {
    "name": "questframe_fh6vr_preflight",
    "description": "Run the FH6VR C# launcher preflight through the Hermes bridge.",
    "parameters": {
        "type": "object",
        "properties": {
            "launcher_exe": {
                "type": "string",
                "description": "Optional one-shot FH6VR.Launcher executable path.",
            },
            "report_path": {
                "type": "string",
                "description": "Optional JSON report path for FH6VR preflight output.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 300,
                "description": "Process timeout.",
            },
        },
    },
}

RTX3060_PROFILES_SCHEMA = {
    "name": "questframe_rtx3060_profiles",
    "description": "Print the FH6VR RTX 3060 DIBR runtime profiles.",
    "parameters": {
        "type": "object",
        "properties": {
            "launcher_exe": {
                "type": "string",
                "description": "Optional one-shot FH6VR.Launcher executable path.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 300,
                "description": "Process timeout.",
            },
        },
    },
}

RTX3060_SELFTEST_SCHEMA = {
    "name": "questframe_rtx3060_selftest",
    "description": "Validate the FH6VR RTX 3060 DIBR profile contract.",
    "parameters": {
        "type": "object",
        "properties": {
            "launcher_exe": {
                "type": "string",
                "description": "Optional one-shot FH6VR.Launcher executable path.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 300,
                "description": "Process timeout.",
            },
        },
    },
}

SESSION_READINESS_SCHEMA = {
    "name": "questframe_session_readiness",
    "description": "Run the FH6VR OpenXR session-readiness probe.",
    "parameters": {
        "type": "object",
        "properties": {
            "launcher_exe": {
                "type": "string",
                "description": "Optional one-shot FH6VR.Launcher executable path.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 300,
                "description": "Process timeout.",
            },
        },
    },
}

GRAPHICS_SESSION_SCHEMA = {
    "name": "questframe_graphics_session",
    "description": "Run the FH6VR OpenXR D3D11 graphics-bound session and swapchain-format probe.",
    "parameters": {
        "type": "object",
        "properties": {
            "launcher_exe": {
                "type": "string",
                "description": "Optional one-shot FH6VR.Launcher executable path.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 300,
                "description": "Process timeout.",
            },
        },
    },
}

FRAME_LOOP_SCHEMA = {
    "name": "questframe_frame_loop",
    "description": "Run the FH6VR minimal OpenXR frame-loop probe.",
    "parameters": {
        "type": "object",
        "properties": {
            "launcher_exe": {
                "type": "string",
                "description": "Optional one-shot FH6VR.Launcher executable path.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 300,
                "description": "Process timeout.",
            },
        },
    },
}

DIBR_SWAPCHAIN_SCHEMA = {
    "name": "questframe_dibr_swapchain",
    "description": "Run the FH6VR DIBR-to-OpenXR swapchain write probe.",
    "parameters": {
        "type": "object",
        "properties": {
            "launcher_exe": {
                "type": "string",
                "description": "Optional one-shot FH6VR.Launcher executable path.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 300,
                "description": "Process timeout.",
            },
        },
    },
}

FH6_CAPTURE_PREFLIGHT_SCHEMA = {
    "name": "questframe_fh6_capture_preflight",
    "description": "Run the FH6VR non-invasive FH6/D3D12 capture preflight.",
    "parameters": {
        "type": "object",
        "properties": {
            "launcher_exe": {
                "type": "string",
                "description": "Optional one-shot FH6VR.Launcher executable path.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 300,
                "description": "Process timeout.",
            },
        },
    },
}

LIVE_CAPTURE_SELFTEST_SCHEMA = {
    "name": "questframe_live_capture_selftest",
    "description": "Run the FH6VR live D3D12/window color capture self-test (0.14 gate).",
    "parameters": {
        "type": "object",
        "properties": {
            "launcher_exe": {
                "type": "string",
                "description": "Optional one-shot FH6VR.Launcher executable path.",
            },
            "attempt_window_capture": {
                "type": "boolean",
                "description": "When true, pass --attempt-window-capture to probe external color read.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 300,
                "description": "Process timeout.",
            },
        },
    },
}

DEPTH_SURFACE_SELFTEST_SCHEMA = {
    "name": "questframe_depth_surface_selftest",
    "description": "Run the FH6VR approved D3D12 depth surface contract self-test (0.15 gate).",
    "parameters": {
        "type": "object",
        "properties": {
            "launcher_exe": {
                "type": "string",
                "description": "Optional one-shot FH6VR.Launcher executable path.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 300,
                "description": "Process timeout.",
            },
        },
    },
}

DEPTH_READER_SELFTEST_SCHEMA = {
    "name": "questframe_depth_reader_selftest",
    "description": "Run the FH6VR approved D3D12 depth reader self-test (0.16 gate).",
    "parameters": {
        "type": "object",
        "properties": {
            "launcher_exe": {
                "type": "string",
                "description": "Optional one-shot FH6VR.Launcher executable path.",
            },
            "fixture": {
                "type": "boolean",
                "description": "When true, pass --fixture to create and read a local shared D3D12 depth surface.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 300,
                "description": "Process timeout.",
            },
        },
    },
}

DEPTH_PRODUCER_SELFTEST_SCHEMA = {
    "name": "questframe_depth_producer_selftest",
    "description": "Run the FH6VR approved D3D12 depth producer metadata self-test (0.17 gate).",
    "parameters": {
        "type": "object",
        "properties": {
            "launcher_exe": {
                "type": "string",
                "description": "Optional one-shot FH6VR.Launcher executable path.",
            },
            "fixture": {
                "type": "boolean",
                "description": "When true, pass --fixture to export fixture metadata and prove reader/DIBR handoff.",
            },
            "metadata_path": {
                "type": "string",
                "description": "Optional path to a producer metadata JSON file.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 300,
                "description": "Process timeout.",
            },
        },
    },
}

COMPANION_DEPTH_PRODUCER_SELFTEST_SCHEMA = {
    "name": "questframe_companion_depth_producer_selftest",
    "description": "Run the FH6VR companion depth producer handoff self-test (0.18 gate).",
    "parameters": {
        "type": "object",
        "properties": {
            "launcher_exe": {
                "type": "string",
                "description": "Optional one-shot FH6VR.Launcher executable path.",
            },
            "approve": {
                "type": "boolean",
                "description": "When true, pass --approve to export shared depth outside the FH6 install folder.",
            },
            "metadata_path": {
                "type": "string",
                "description": "Optional path for companion producer metadata JSON output.",
            },
            "output_dir": {
                "type": "string",
                "description": "Optional export directory outside the FH6 install folder.",
            },
            "frames": {
                "type": "integer",
                "minimum": 1,
                "maximum": 120,
                "description": "Number of cadence frames to export when approved.",
            },
            "interval_ms": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
                "description": "Target interval between exported frames in milliseconds.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 300,
                "description": "Process timeout.",
            },
        },
    },
}

COLOR_DEPTH_PAIRING_SELFTEST_SCHEMA = {
    "name": "questframe_color_depth_pairing_selftest",
    "description": "Run the FH6VR live color + companion depth pairing self-test (0.19 gate).",
    "parameters": {
        "type": "object",
        "properties": {
            "launcher_exe": {
                "type": "string",
                "description": "Optional one-shot FH6VR.Launcher executable path.",
            },
            "approve": {
                "type": "boolean",
                "description": "When true, pass --approve for companion depth export.",
            },
            "attempt_window_capture": {
                "type": "boolean",
                "description": "When true, pass --attempt-window-capture for live FH6 color.",
            },
            "metadata_path": {
                "type": "string",
                "description": "Optional companion producer metadata JSON path.",
            },
            "output_dir": {
                "type": "string",
                "description": "Optional companion export directory.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 300,
                "description": "Process timeout.",
            },
        },
    },
}

OPENXR_PRESENTATION_SELFTEST_SCHEMA = {
    "name": "questframe_openxr_presentation_selftest",
    "description": "Run the FH6VR OpenXR presentation self-test with optional color/depth pairing (0.19 gate).",
    "parameters": {
        "type": "object",
        "properties": {
            "launcher_exe": {
                "type": "string",
                "description": "Optional one-shot FH6VR.Launcher executable path.",
            },
            "approve": {
                "type": "boolean",
                "description": "When true, pass --approve to run pairing before presentation.",
            },
            "attempt_window_capture": {
                "type": "boolean",
                "description": "When true, pass --attempt-window-capture during pairing.",
            },
            "require_pairing": {
                "type": "boolean",
                "description": "When true, pass --require-pairing and fail if pairing did not pass.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 300,
                "description": "Process timeout.",
            },
        },
    },
}

SUPPORT_REPORT_SCHEMA = {
    "name": "questframe_support_report",
    "description": "Create redacted QuestFrame/FH6VR JSON and HTML support reports.",
    "parameters": {
        "type": "object",
        "properties": {
            "launcher_exe": {
                "type": "string",
                "description": "Optional one-shot FH6VR.Launcher executable path.",
            },
            "json_path": {
                "type": "string",
                "description": "Optional output path for the redacted JSON report.",
            },
            "html_path": {
                "type": "string",
                "description": "Optional output path for the redacted HTML report.",
            },
            "include_live_openxr": {
                "type": "boolean",
                "description": "Include live OpenXR session, frame-loop, and DIBR swapchain probes.",
            },
            "no_openxr": {
                "type": "boolean",
                "description": "Skip OpenXR probes and create a reduced offline report.",
            },
            "include_sensitive_paths": {
                "type": "boolean",
                "description": "Local-only debugging option. Do not enable for BOOTH support exports.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 300,
                "description": "Process timeout.",
            },
        },
    },
}

UNITY_SCAN_SCHEMA = {
    "name": "questframe_unity_scan",
    "description": "Read-only scan of Unity/VCC project packages for VRChat tool risk.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_path": {
                "type": "string",
                "description": "Optional Unity project path. If omitted, configured roots are scanned.",
            },
            "max_projects": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "description": "Maximum projects to inspect when scanning roots.",
            },
        },
    },
}


@dataclass
class LauncherRun:
    ok: bool
    returncode: int
    command: list[str]
    stdout: str
    stderr: str
    parsed: Any


def check_available() -> bool:
    return True


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_config_readonly() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _plugin_config() -> dict[str, Any]:
    plugins = _load_config_readonly().get("plugins", {})
    if not isinstance(plugins, dict):
        return {}
    entries = plugins.get("entries", {})
    if not isinstance(entries, dict):
        return {}
    for key in CONFIG_ALIASES:
        value = entries.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def save_setup_values(values: dict[str, Any]) -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config, save_config
    except Exception as exc:
        return {"ok": False, "error": f"Hermes config writer unavailable: {exc}"}

    cfg = load_config()
    plugins = cfg.setdefault("plugins", {})
    if not isinstance(plugins, dict):
        plugins = {}
        cfg["plugins"] = plugins
    entries = plugins.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        plugins["entries"] = entries
    entry = entries.setdefault(PLUGIN_ID, {})
    if not isinstance(entry, dict):
        entry = {}
        entries[PLUGIN_ID] = entry

    saved: list[str] = []
    launcher = str(values.get("launcher_exe") or "").strip()
    if launcher:
        entry["launcher_exe"] = launcher
        saved.append("launcher_exe")

    unity_python = str(values.get("unity_python") or "").strip()
    if unity_python:
        entry["unity_python"] = unity_python
        saved.append("unity_python")

    roots = values.get("vcc_project_roots")
    if isinstance(roots, list):
        clean_roots = [str(root).strip() for root in roots if str(root).strip()]
        if clean_roots:
            entry["vcc_project_roots"] = clean_roots
            saved.append("vcc_project_roots")

    save_config(cfg)
    return {"ok": True, "saved": saved, "config_key": f"plugins.entries.{PLUGIN_ID}"}


def _path_or_empty(value: Any) -> str:
    return str(value or "").strip()


def resolve_launcher_path(explicit: str | None = None) -> Path | None:
    cfg = _plugin_config()
    candidates = [
        explicit,
        cfg.get("launcher_exe"),
        cfg.get("launcher_path"),
        os.environ.get("QUESTFRAME_FH6VR_EXE"),
        os.environ.get("FH6VR_LAUNCHER_EXE"),
    ]
    for raw in candidates:
        text = _path_or_empty(raw)
        if not text:
            continue
        return Path(text).expanduser()
    return None


def _bounded(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[TRUNCATED]"


def _report_dir() -> Path:
    path = get_hermes_home() / "reports" / "questframe"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_report_paths(prefix: str = "questframe-support") -> tuple[Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = _report_dir() / f"{prefix}-{stamp}"
    return base.with_suffix(".json"), base.with_suffix(".html")


def _parse_json_stdout(stdout: str) -> Any:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def run_launcher(
    command: str,
    *,
    launcher_exe: str | None = None,
    extra_args: list[str] | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    launcher = resolve_launcher_path(launcher_exe)
    if launcher is None:
        return {
            "ok": False,
            "error": "FH6VR launcher path is not configured.",
            "configure": f"hermes questframe setup --launcher-exe <path-to-FH6VR.Launcher>",
        }
    if not launcher.exists():
        return {
            "ok": False,
            "error": "FH6VR launcher path does not exist.",
            "launcher_exe": str(launcher),
        }

    timeout = max(5, min(int(timeout_seconds or DEFAULT_TIMEOUT_SECONDS), 300))
    argv = [str(launcher), command]
    argv.extend(extra_args or [])
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"FH6VR launcher timed out after {timeout}s.",
            "launcher_exe": str(launcher),
            "command": argv,
            "stdout": _bounded(exc.stdout or ""),
            "stderr": _bounded(exc.stderr or ""),
        }
    except OSError as exc:
        return {
            "ok": False,
            "error": f"Could not start FH6VR launcher: {exc}",
            "launcher_exe": str(launcher),
            "command": argv,
        }

    parsed = _parse_json_stdout(completed.stdout or "")
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "launcher_exe": str(launcher),
        "command": argv,
        "stdout": _bounded(completed.stdout or ""),
        "stderr": _bounded(completed.stderr or ""),
        "json": parsed,
    }


def _vcc_roots() -> list[Path]:
    cfg = _plugin_config()
    configured = cfg.get("vcc_project_roots")
    roots: list[Path] = []
    if isinstance(configured, list):
        roots.extend(Path(str(item)).expanduser() for item in configured if str(item).strip())

    user_profile = Path(os.environ.get("USERPROFILE") or str(Path.home()))
    roots.extend(
        [
            user_profile / "Documents" / "VRChat Projects",
            user_profile / "Documents" / "Unity",
        ]
    )
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _is_unity_project(path: Path) -> bool:
    return (path / "Assets").is_dir() and (path / "Packages" / "manifest.json").is_file()


def _candidate_projects(root: Path, max_projects: int) -> list[Path]:
    if _is_unity_project(root):
        return [root]
    if not root.is_dir():
        return []
    found: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if child.is_dir() and _is_unity_project(child):
            found.append(child)
            if len(found) >= max_projects:
                break
    return found


def _read_project_version(project: Path) -> str:
    version_path = project / "ProjectSettings" / "ProjectVersion.txt"
    if not version_path.exists():
        return ""
    try:
        for line in version_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("m_EditorVersion:"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return ""
    return ""


def _read_package_manifest(project: Path) -> dict[str, str]:
    manifest_path = project / "Packages" / "manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    deps = raw.get("dependencies", {})
    if not isinstance(deps, dict):
        return {}
    return {str(key): str(value) for key, value in deps.items()}


def _limited_unitypackage_markers(project: Path, limit: int = 20) -> list[str]:
    assets = project / "Assets"
    markers: list[str] = []
    if not assets.is_dir():
        return markers
    try:
        iterator = assets.rglob("*.unitypackage")
        for path in iterator:
            markers.append(str(path))
            if len(markers) >= limit:
                break
    except OSError:
        return markers
    return markers


def scan_unity_project(project: Path) -> dict[str, Any]:
    packages = _read_package_manifest(project)
    detected = [
        {
            "id": package_id,
            "name": KNOWN_VPM_PACKAGES[package_id],
            "version": packages.get(package_id, ""),
        }
        for package_id in sorted(KNOWN_VPM_PACKAGES)
        if package_id in packages
    ]
    sdk_avatar = "com.vrchat.avatars" in packages
    sdk_world = "com.vrchat.worlds" in packages
    risks: list[str] = []
    if not packages:
        risks.append("package manifest missing or unreadable")
    if not sdk_avatar and not sdk_world:
        risks.append("VRChat SDK package not detected")
    if sdk_avatar and sdk_world:
        risks.append("both Avatar and World SDK packages detected")
    markers = _limited_unitypackage_markers(project)
    if markers:
        risks.append("legacy .unitypackage files are present under Assets")

    return {
        "project_path": str(project),
        "exists": project.exists(),
        "unity_project": _is_unity_project(project),
        "unity_version": _read_project_version(project),
        "package_count": len(packages),
        "detected_packages": detected,
        "risks": risks,
        "unitypackage_markers": markers,
    }


def scan_unity_projects(
    *, project_path: str | None = None, max_projects: int | None = None
) -> dict[str, Any]:
    limit = max(1, min(int(max_projects or 25), 100))
    if project_path:
        projects = [Path(project_path).expanduser()]
        roots = []
    else:
        roots = _vcc_roots()
        projects = []
        for root in roots:
            remaining = limit - len(projects)
            if remaining <= 0:
                break
            projects.extend(_candidate_projects(root, remaining))

    scanned = [scan_unity_project(project) for project in projects[:limit]]
    return {
        "ok": True,
        "scanned_at": _now_utc(),
        "roots": [str(root) for root in roots],
        "project_count": len(scanned),
        "projects": scanned,
    }


def status() -> dict[str, Any]:
    cfg = _plugin_config()
    launcher = resolve_launcher_path()
    return {
        "ok": True,
        "plugin": PLUGIN_NAME,
        "hermes_home": str(get_hermes_home()),
        "config_key": f"plugins.entries.{PLUGIN_ID}",
        "configured": bool(cfg),
        "launcher_exe": str(launcher) if launcher else "",
        "launcher_exists": bool(launcher and launcher.exists()),
        "csharp_bridge_configured": bool(launcher and launcher.exists()),
        "python_unity_bridge": {
            "unity_python": str(cfg.get("unity_python") or ""),
            "vcc_project_roots": cfg.get("vcc_project_roots") or [],
        },
        "available_tools": [
            "questframe_status",
            "questframe_setup",
            "questframe_fh6vr_preflight",
            "questframe_rtx3060_profiles",
            "questframe_rtx3060_selftest",
            "questframe_session_readiness",
            "questframe_graphics_session",
            "questframe_frame_loop",
            "questframe_dibr_swapchain",
            "questframe_fh6_capture_preflight",
            "questframe_live_capture_selftest",
            "questframe_depth_surface_selftest",
            "questframe_depth_reader_selftest",
            "questframe_depth_producer_selftest",
            "questframe_companion_depth_producer_selftest",
            "questframe_color_depth_pairing_selftest",
            "questframe_openxr_presentation_selftest",
            "questframe_support_report",
            "questframe_unity_scan",
        ],
        "next_step": (
            "Run questframe_setup with launcher_exe before bridge probes."
            if not launcher
            else "Run questframe_support_report or questframe_rtx3060_selftest."
        ),
    }


def handle_status(args: dict[str, Any] | None = None, **_: Any) -> str:
    return _json(status())


def handle_setup(args: dict[str, Any] | None = None, **_: Any) -> str:
    return _json(save_setup_values(args or {}))


def handle_preflight(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    extra = ["--json"]
    report_path = str(args.get("report_path") or "").strip()
    if report_path:
        extra.extend(["--write-report", report_path])
    return _json(
        run_launcher(
            "preflight",
            launcher_exe=str(args.get("launcher_exe") or "") or None,
            extra_args=extra,
            timeout_seconds=int(args.get("timeout_seconds") or 0) or None,
        )
    )


def handle_rtx3060_profiles(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(
        run_launcher(
            "profiles",
            launcher_exe=str(args.get("launcher_exe") or "") or None,
            extra_args=["--json"],
            timeout_seconds=int(args.get("timeout_seconds") or 0) or None,
        )
    )


def handle_rtx3060_selftest(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(
        run_launcher(
            "rtx3060-selftest",
            launcher_exe=str(args.get("launcher_exe") or "") or None,
            extra_args=["--json"],
            timeout_seconds=int(args.get("timeout_seconds") or 0) or None,
        )
    )


def handle_session_readiness(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(
        run_launcher(
            "session-readiness-selftest",
            launcher_exe=str(args.get("launcher_exe") or "") or None,
            extra_args=["--json"],
            timeout_seconds=int(args.get("timeout_seconds") or 0) or None,
        )
    )


def handle_graphics_session(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(
        run_launcher(
            "graphics-session-selftest",
            launcher_exe=str(args.get("launcher_exe") or "") or None,
            extra_args=["--json"],
            timeout_seconds=int(args.get("timeout_seconds") or 0) or None,
        )
    )


def handle_frame_loop(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(
        run_launcher(
            "frame-loop-selftest",
            launcher_exe=str(args.get("launcher_exe") or "") or None,
            extra_args=["--json"],
            timeout_seconds=int(args.get("timeout_seconds") or 0) or None,
        )
    )


def handle_dibr_swapchain(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(
        run_launcher(
            "dibr-swapchain-selftest",
            launcher_exe=str(args.get("launcher_exe") or "") or None,
            extra_args=["--json"],
            timeout_seconds=int(args.get("timeout_seconds") or 0) or None,
        )
    )


def handle_fh6_capture_preflight(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(
        run_launcher(
            "fh6-capture-preflight",
            launcher_exe=str(args.get("launcher_exe") or "") or None,
            extra_args=["--json"],
            timeout_seconds=int(args.get("timeout_seconds") or 0) or None,
        )
    )


def handle_live_capture_selftest(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    extra = ["--json"]
    if bool(args.get("attempt_window_capture")):
        extra.append("--attempt-window-capture")
    return _json(
        run_launcher(
            "fh6-live-capture-selftest",
            launcher_exe=str(args.get("launcher_exe") or "") or None,
            extra_args=extra,
            timeout_seconds=int(args.get("timeout_seconds") or 0) or None,
        )
    )


def handle_depth_surface_selftest(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(
        run_launcher(
            "fh6-depth-surface-selftest",
            launcher_exe=str(args.get("launcher_exe") or "") or None,
            extra_args=["--json"],
            timeout_seconds=int(args.get("timeout_seconds") or 0) or None,
        )
    )


def handle_depth_reader_selftest(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    extra = ["--json"]
    if bool(args.get("fixture")):
        extra.append("--fixture")
    return _json(
        run_launcher(
            "fh6-depth-reader-selftest",
            launcher_exe=str(args.get("launcher_exe") or "") or None,
            extra_args=extra,
            timeout_seconds=int(args.get("timeout_seconds") or 0) or None,
        )
    )


def handle_depth_producer_selftest(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    extra = ["--json"]
    if bool(args.get("fixture")):
        extra.append("--fixture")
    metadata_path = str(args.get("metadata_path") or "").strip()
    if metadata_path:
        extra.extend(["--metadata", metadata_path])
    return _json(
        run_launcher(
            "fh6-depth-producer-selftest",
            launcher_exe=str(args.get("launcher_exe") or "") or None,
            extra_args=extra,
            timeout_seconds=int(args.get("timeout_seconds") or 0) or None,
        )
    )


def handle_companion_depth_producer_selftest(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    extra = ["--json"]
    if bool(args.get("approve")):
        extra.append("--approve")
    metadata_path = str(args.get("metadata_path") or "").strip()
    if metadata_path:
        extra.extend(["--metadata", metadata_path])
    output_dir = str(args.get("output_dir") or "").strip()
    if output_dir:
        extra.extend(["--output-dir", output_dir])
    frames = int(args.get("frames") or 0)
    if frames > 0:
        extra.extend(["--frames", str(frames)])
    interval_ms = int(args.get("interval_ms") or 0)
    if interval_ms > 0:
        extra.extend(["--interval-ms", str(interval_ms)])
    return _json(
        run_launcher(
            "fh6-companion-depth-producer-selftest",
            launcher_exe=str(args.get("launcher_exe") or "") or None,
            extra_args=extra,
            timeout_seconds=int(args.get("timeout_seconds") or 0) or None,
        )
    )


def handle_color_depth_pairing_selftest(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    extra = ["--json"]
    if bool(args.get("approve")):
        extra.append("--approve")
    if bool(args.get("attempt_window_capture")):
        extra.append("--attempt-window-capture")
    metadata_path = str(args.get("metadata_path") or "").strip()
    if metadata_path:
        extra.extend(["--metadata", metadata_path])
    output_dir = str(args.get("output_dir") or "").strip()
    if output_dir:
        extra.extend(["--output-dir", output_dir])
    return _json(
        run_launcher(
            "fh6-color-depth-pairing-selftest",
            launcher_exe=str(args.get("launcher_exe") or "") or None,
            extra_args=extra,
            timeout_seconds=int(args.get("timeout_seconds") or 0) or None,
        )
    )


def handle_openxr_presentation_selftest(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    extra = ["--json"]
    if bool(args.get("approve")):
        extra.append("--approve")
    if bool(args.get("attempt_window_capture")):
        extra.append("--attempt-window-capture")
    if bool(args.get("require_pairing")):
        extra.append("--require-pairing")
    return _json(
        run_launcher(
            "openxr-presentation-selftest",
            launcher_exe=str(args.get("launcher_exe") or "") or None,
            extra_args=extra,
            timeout_seconds=int(args.get("timeout_seconds") or 0) or None,
        )
    )


def support_report(
    *,
    launcher_exe: str | None = None,
    json_path: str | None = None,
    html_path: str | None = None,
    include_live_openxr: bool = False,
    no_openxr: bool = False,
    include_sensitive_paths: bool = False,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    if not json_path or not html_path:
        default_json, default_html = _default_report_paths(
            "questframe-support-live" if include_live_openxr else "questframe-support"
        )
        json_path = json_path or str(default_json)
        html_path = html_path or str(default_html)

    extra = [
        "--json",
        "--write-json",
        str(Path(json_path).expanduser()),
        "--write-html",
        str(Path(html_path).expanduser()),
    ]
    if include_live_openxr:
        extra.append("--include-live-openxr")
    if no_openxr:
        extra.append("--no-openxr")
    if include_sensitive_paths:
        extra.append("--include-sensitive-paths")

    result = run_launcher(
        "support-report",
        launcher_exe=launcher_exe,
        extra_args=extra,
        timeout_seconds=timeout_seconds,
    )
    result["report_paths"] = {
        "json": str(Path(json_path).expanduser()),
        "html": str(Path(html_path).expanduser()),
    }
    result["redacted_by_default"] = not include_sensitive_paths
    return result


def handle_support_report(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(
        support_report(
            launcher_exe=str(args.get("launcher_exe") or "") or None,
            json_path=str(args.get("json_path") or "") or None,
            html_path=str(args.get("html_path") or "") or None,
            include_live_openxr=bool(args.get("include_live_openxr")),
            no_openxr=bool(args.get("no_openxr")),
            include_sensitive_paths=bool(args.get("include_sensitive_paths")),
            timeout_seconds=int(args.get("timeout_seconds") or 0) or None,
        )
    )


def handle_unity_scan(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(
        scan_unity_projects(
            project_path=str(args.get("project_path") or "") or None,
            max_projects=int(args.get("max_projects") or 0) or None,
        )
    )


HELP = """questframe commands:
  /questframe status
  /questframe preflight
  /questframe profiles
  /questframe rtx3060-selftest
  /questframe session
  /questframe graphics-session
  /questframe frame-loop
  /questframe dibr-swapchain
  /questframe capture-preflight
  /questframe live-capture-selftest
  /questframe depth-surface-selftest
  /questframe depth-reader-selftest [--fixture]
  /questframe depth-producer-selftest [--fixture] [--metadata PATH]
  /questframe companion-depth-producer-selftest [--approve] [--metadata PATH] [--output-dir PATH]
  /questframe color-depth-pairing-selftest [--approve] [--attempt-window-capture]
  /questframe openxr-presentation-selftest [--approve] [--require-pairing]
  /questframe support-report
  /questframe unity-scan [project_path]
"""


def handle_slash(raw_args: str) -> str:
    argv = (raw_args or "").strip().split()
    if not argv or argv[0] in {"help", "-h", "--help"}:
        return HELP
    command = argv[0].lower()
    if command == "status":
        return _json(status())
    if command == "preflight":
        return handle_preflight({})
    if command in {"profiles", "rtx3060-profiles"}:
        return handle_rtx3060_profiles({})
    if command in {"rtx3060-selftest", "rtx3060"}:
        return handle_rtx3060_selftest({})
    if command in {"session", "session-readiness"}:
        return handle_session_readiness({})
    if command in {"graphics-session", "graphics"}:
        return handle_graphics_session({})
    if command in {"frame-loop", "frame"}:
        return handle_frame_loop({})
    if command in {"dibr-swapchain", "swapchain"}:
        return handle_dibr_swapchain({})
    if command in {"capture-preflight", "fh6-capture-preflight", "capture"}:
        return handle_fh6_capture_preflight({})
    if command in {"live-capture-selftest", "live-capture", "live-capture-self-test"}:
        return handle_live_capture_selftest({})
    if command in {"depth-surface-selftest", "depth-surface", "depth"}:
        return handle_depth_surface_selftest({})
    if command in {"depth-reader-selftest", "depth-reader", "reader"}:
        return handle_depth_reader_selftest({"fixture": "--fixture" in argv})
    if command in {"depth-producer-selftest", "depth-producer", "producer"}:
        args: dict[str, Any] = {"fixture": "--fixture" in argv}
        if "--metadata" in argv:
            index = argv.index("--metadata")
            if index + 1 < len(argv):
                args["metadata_path"] = argv[index + 1]
        return handle_depth_producer_selftest(args)
    if command in {
        "companion-depth-producer-selftest",
        "companion-depth-producer",
        "companion-producer",
    }:
        args = {"approve": "--approve" in argv}
        if "--metadata" in argv:
            index = argv.index("--metadata")
            if index + 1 < len(argv):
                args["metadata_path"] = argv[index + 1]
        if "--output-dir" in argv:
            index = argv.index("--output-dir")
            if index + 1 < len(argv):
                args["output_dir"] = argv[index + 1]
        return handle_companion_depth_producer_selftest(args)
    if command in {
        "color-depth-pairing-selftest",
        "color-depth-pairing",
        "pairing-selftest",
    }:
        args = {
            "approve": "--approve" in argv,
            "attempt_window_capture": "--attempt-window-capture" in argv,
        }
        if "--metadata" in argv:
            index = argv.index("--metadata")
            if index + 1 < len(argv):
                args["metadata_path"] = argv[index + 1]
        if "--output-dir" in argv:
            index = argv.index("--output-dir")
            if index + 1 < len(argv):
                args["output_dir"] = argv[index + 1]
        return handle_color_depth_pairing_selftest(args)
    if command in {
        "openxr-presentation-selftest",
        "openxr-presentation",
        "presentation-selftest",
    }:
        return handle_openxr_presentation_selftest(
            {
                "approve": "--approve" in argv,
                "attempt_window_capture": "--attempt-window-capture" in argv,
                "require_pairing": "--require-pairing" in argv,
            }
        )
    if command in {"support-report", "report"}:
        return handle_support_report({})
    if command == "unity-scan":
        project = argv[1] if len(argv) > 1 else None
        return _json(scan_unity_projects(project_path=project))
    return f"Unknown questframe command: {command}\n\n{HELP}"

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from hermes_constants import get_hermes_home
except Exception:
    def get_hermes_home() -> Path:  # type: ignore[no-redef]
        return Path.home() / ".hermes"


PLUGIN_ID = "unity-vrchat-bridge"
PLUGIN_NAME = "unity-vrchat-bridge"
SUPPORTED_VRCHAT_UNITY_VERSION = "2022.3.22f1"
DEFAULT_PORT = 17751
SESSION_RELATIVE_PATH = Path("Library") / "HermesUnityBridge" / "session.json"
MAX_ARCHIVE_ENTRIES = 5000

KNOWN_VPM_PACKAGES = {
    "com.vrchat.base": "VRChat SDK Base",
    "com.vrchat.avatars": "VRChat SDK Avatars",
    "com.vrchat.worlds": "VRChat SDK Worlds",
    "nadena.dev.modular-avatar": "Modular Avatar",
    "nadena.dev.ndmf": "NDMF",
    "jp.lilxyzw.liltoon": "lilToon",
    "com.vrcfury.vrcfury": "VRCFury",
    "com.anatawa12.avatar-optimizer": "Avatar Optimizer",
    "com.poiyomi.toon": "Poiyomi Toon",
    "com.poiyomi.shader": "Poiyomi Shader",
}

DEPENDENCY_HINTS = {
    "liltoon": ("lilToon", "jp.lilxyzw.liltoon"),
    "lil": ("lilToon", "jp.lilxyzw.liltoon"),
    "modular avatar": ("Modular Avatar", "nadena.dev.modular-avatar"),
    "modularavatar": ("Modular Avatar", "nadena.dev.modular-avatar"),
    "ndmf": ("NDMF", "nadena.dev.ndmf"),
    "vrcfury": ("VRCFury", "com.vrcfury.vrcfury"),
    "poiyomi": ("Poiyomi", "com.poiyomi.toon"),
}

EDITOR_LOG_RULES = (
    ("unity.compile.error", "error", re.compile(r"error CS\d+:|Assembly .* failed", re.I)),
    ("vrc.sdk.validation", "error", re.compile(r"Validation failed|VRCSDK", re.I)),
    (
        "missing.script",
        "error",
        re.compile(r"The referenced script on this Behaviour is missing|Missing MonoBehaviour", re.I),
    ),
    ("shader.missing", "warning", re.compile(r"Shader .* not found|Hidden/InternalErrorShader", re.I)),
    ("guid.meta.conflict", "error", re.compile(r"GUID \[.*\] for asset|meta file", re.I)),
)


STATUS_SCHEMA = {
    "name": "unity_bridge_status",
    "description": "Show Unity/VRChat bridge readiness without mutating Unity projects.",
    "parameters": {"type": "object", "properties": {}},
}

HEALTH_SCHEMA = {
    "name": "unity_bridge_health",
    "description": "Call a trusted project-local Unity bridge /health endpoint.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_path": {
                "type": "string",
                "description": "Unity project path containing Library/HermesUnityBridge/session.json.",
            },
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 30,
                "description": "HTTP timeout for the local bridge.",
            },
        },
        "required": ["project_path"],
    },
}

SNAPSHOT_SCHEMA = {
    "name": "unity_bridge_snapshot",
    "description": "Call a trusted project-local Unity bridge /snapshot endpoint.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_path": {"type": "string", "description": "Unity project path."},
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 30,
                "description": "HTTP timeout for the local bridge.",
            },
        },
        "required": ["project_path"],
    },
}

SELECTION_SCHEMA = {
    "name": "unity_bridge_selection_get",
    "description": "Call a trusted Unity bridge /selection endpoint.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_path": {"type": "string", "description": "Unity project path."},
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 30,
                "description": "HTTP timeout for the local bridge.",
            },
        },
        "required": ["project_path"],
    },
}

CAPABILITIES_SCHEMA = {
    "name": "unity_bridge_capabilities",
    "description": "List the trusted Unity Editor bridge capabilities and safety policy.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_path": {"type": "string", "description": "Unity project path."},
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 30,
                "description": "HTTP timeout for the local bridge.",
            },
        },
        "required": ["project_path"],
    },
}

PACKAGES_SCHEMA = {
    "name": "unity_bridge_packages",
    "description": "Read package metadata from a trusted live Unity Editor bridge.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_path": {"type": "string", "description": "Unity project path."},
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 30,
                "description": "HTTP timeout for the local bridge.",
            },
        },
        "required": ["project_path"],
    },
}

HIERARCHY_SCHEMA = {
    "name": "unity_bridge_scene_hierarchy",
    "description": "Read the active scene hierarchy from a trusted live Unity Editor bridge.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_path": {"type": "string", "description": "Unity project path."},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 500,
                "description": "Maximum GameObjects to return.",
            },
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 30,
                "description": "HTTP timeout for the local bridge.",
            },
        },
        "required": ["project_path"],
    },
}

CONSOLE_RECENT_SCHEMA = {
    "name": "unity_bridge_console_recent",
    "description": "Call a trusted Unity bridge /logs/recent endpoint.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_path": {"type": "string", "description": "Unity project path."},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 500,
                "description": "Maximum recent log entries.",
            },
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 30,
                "description": "HTTP timeout for the local bridge.",
            },
        },
        "required": ["project_path"],
    },
}

ASSET_SEARCH_SCHEMA = {
    "name": "unity_bridge_asset_search",
    "description": "Call a trusted Unity bridge /assets/search endpoint.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_path": {"type": "string", "description": "Unity project path."},
            "filter": {"type": "string", "description": "Unity AssetDatabase filter, such as t:Prefab."},
            "folders": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Asset folders to search. Defaults to Assets.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 500,
                "description": "Maximum asset results.",
            },
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 30,
                "description": "HTTP timeout for the local bridge.",
            },
        },
        "required": ["project_path", "filter"],
    },
}

ASSET_INFO_SCHEMA = {
    "name": "unity_bridge_asset_info",
    "description": "Inspect Unity asset metadata and optional dependencies through the live Editor bridge.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_path": {"type": "string", "description": "Unity project path."},
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Unity asset paths, such as Assets/My.prefab.",
            },
            "include_dependencies": {
                "type": "boolean",
                "description": "Include direct Unity asset dependencies.",
            },
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 30,
                "description": "HTTP timeout for the local bridge.",
            },
        },
        "required": ["project_path", "paths"],
    },
}

MENU_EXECUTE_SCHEMA = {
    "name": "unity_bridge_menu_execute",
    "description": "Plan an allowlisted Unity menu command; MVP rejects live execution.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_path": {"type": "string", "description": "Unity project path."},
            "menu_path": {"type": "string", "description": "Unity menu path."},
            "dry_run": {
                "type": "boolean",
                "description": "Must be true in the MVP.",
            },
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 30,
                "description": "HTTP timeout for the local bridge.",
            },
        },
        "required": ["project_path", "menu_path"],
    },
}

OPERATION_PLAN_SCHEMA = {
    "name": "unity_bridge_operation_plan",
    "description": "Plan a generic Unity Editor operation; MVP rejects live execution.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_path": {"type": "string", "description": "Unity project path."},
            "operation": {"type": "string", "description": "Operation name, such as asset_create."},
            "target_path": {"type": "string", "description": "Optional target asset path under Assets."},
            "dry_run": {
                "type": "boolean",
                "description": "Must be true in the MVP.",
            },
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 30,
                "description": "HTTP timeout for the local bridge.",
            },
        },
        "required": ["project_path", "operation"],
    },
}

PLAN_APPLY_SCHEMA = {
    "name": "unity_bridge_plan_apply",
    "description": "Submit a Unity bridge plan for dry-run review; MVP rejects live apply.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_path": {"type": "string", "description": "Unity project path."},
            "operation": {"type": "string", "description": "Plan operation name."},
            "dry_run": {
                "type": "boolean",
                "description": "Must be true in the MVP.",
            },
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 30,
                "description": "HTTP timeout for the local bridge.",
            },
        },
        "required": ["project_path", "operation"],
    },
}

PROJECT_PROFILE_SCHEMA = {
    "name": "unity_project_profile",
    "description": "Read Unity package and project metadata for a local project.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_path": {"type": "string", "description": "Unity project path."},
        },
        "required": ["project_path"],
    },
}

VRCHAT_PROJECT_HEALTH_SCHEMA = {
    "name": "vrchat_project_health",
    "description": "Read-only VRChat/VCC/VPM project health check for a Unity project.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_path": {"type": "string", "description": "Unity project path."},
        },
        "required": ["project_path"],
    },
}

COMMERCIAL_ASSET_INSPECT_SCHEMA = {
    "name": "commercial_asset_inspect_archive",
    "description": "Inspect a local zip or unitypackage for VRChat import risks without extracting it.",
    "parameters": {
        "type": "object",
        "properties": {
            "archive_path": {"type": "string", "description": "Local .zip or .unitypackage path."},
            "project_path": {
                "type": "string",
                "description": "Optional Unity project path used to compare installed dependencies.",
            },
        },
        "required": ["archive_path"],
    },
}


@dataclass(frozen=True)
class BridgeSession:
    project_path: Path
    project_hash: str
    port: int
    token: str

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


class UnityBridgeError(RuntimeError):
    pass


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def check_available() -> bool:
    return True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_project_version(project: Path) -> str:
    version_file = project / "ProjectSettings" / "ProjectVersion.txt"
    try:
        text = version_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    match = re.search(r"m_EditorVersion:\s*(.+)", text)
    return match.group(1).strip() if match else ""


def _is_unity_project(project: Path) -> bool:
    return (project / "Assets").is_dir() and (project / "ProjectSettings").is_dir()


def _read_package_manifest(project: Path) -> dict[str, str]:
    manifest = _read_json(project / "Packages" / "manifest.json")
    deps = manifest.get("dependencies")
    if not isinstance(deps, dict):
        return {}
    return {str(k): str(v) for k, v in deps.items()}


def _read_vpm_manifest(project: Path) -> dict[str, str]:
    for candidate in (project / "vpm-manifest.json", project / "Packages" / "vpm-manifest.json"):
        data = _read_json(candidate)
        locked = data.get("locked")
        if isinstance(locked, dict):
            return {
                str(pkg): str(info.get("version", ""))
                for pkg, info in locked.items()
                if isinstance(info, dict)
            }
        deps = data.get("dependencies")
        if isinstance(deps, dict):
            return {
                str(pkg): str(info.get("version", info))
                for pkg, info in deps.items()
            }
    return {}


def _read_packages_lock(project: Path) -> dict[str, str]:
    data = _read_json(project / "Packages" / "packages-lock.json")
    deps = data.get("dependencies")
    if not isinstance(deps, dict):
        return {}
    result: dict[str, str] = {}
    for package_id, info in deps.items():
        if isinstance(info, dict):
            result[str(package_id)] = str(info.get("version", ""))
    return result


def _detect_packages(packages: dict[str, str], vpm_locked: dict[str, str]) -> list[dict[str, str]]:
    detected = []
    for package_id in sorted(KNOWN_VPM_PACKAGES):
        if package_id in packages or package_id in vpm_locked:
            detected.append(
                {
                    "id": package_id,
                    "name": KNOWN_VPM_PACKAGES[package_id],
                    "version": vpm_locked.get(package_id) or packages.get(package_id, ""),
                    "source": "vpm-lock" if package_id in vpm_locked else "manifest",
                }
            )
    return detected


def _project_hash(project: Path) -> str:
    resolved = str(project.resolve()).replace("\\", "/").lower()
    return "sha256:" + hashlib.sha256(resolved.encode("utf-8")).hexdigest()


def project_profile(project_path: str) -> dict[str, Any]:
    project = Path(project_path).expanduser()
    packages = _read_package_manifest(project)
    vpm_locked = _read_vpm_manifest(project)
    packages_lock = _read_packages_lock(project)
    sdk_avatar = "com.vrchat.avatars" in packages or "com.vrchat.avatars" in vpm_locked
    sdk_world = "com.vrchat.worlds" in packages or "com.vrchat.worlds" in vpm_locked
    sdk_base = "com.vrchat.base" in packages or "com.vrchat.base" in vpm_locked
    risks: list[str] = []
    if not project.exists():
        risks.append("project path does not exist")
    if project.exists() and not _is_unity_project(project):
        risks.append("path is not a Unity project")
    if not packages:
        risks.append("Packages/manifest.json missing or unreadable")
    if sdk_avatar and sdk_world:
        risks.append("both VRChat Avatar and World SDK packages detected")
    if (sdk_avatar or sdk_world or sdk_base) and not vpm_locked:
        risks.append("VRChat SDK detected but VPM manifest lock is missing or unreadable")

    unity_version = _read_project_version(project)
    return {
        "ok": True,
        "scanned_at": _now_utc(),
        "project_path": str(project),
        "project_hash": _project_hash(project) if project.exists() else "",
        "exists": project.exists(),
        "unity_project": _is_unity_project(project),
        "unity_version": unity_version,
        "vrchat_supported_unity_version": SUPPORTED_VRCHAT_UNITY_VERSION,
        "vrchat_unity_version_match": unity_version == SUPPORTED_VRCHAT_UNITY_VERSION,
        "package_count": len(packages),
        "packages_lock_count": len(packages_lock),
        "vpm_locked_count": len(vpm_locked),
        "detected_packages": _detect_packages(packages, vpm_locked),
        "vrchat_project_kind": (
            "avatar" if sdk_avatar and not sdk_world else
            "world" if sdk_world and not sdk_avatar else
            "mixed" if sdk_avatar and sdk_world else
            "generic_unity"
        ),
        "risks": risks,
    }


def vrchat_project_health(project_path: str) -> dict[str, Any]:
    profile = project_profile(project_path)
    project = Path(project_path).expanduser()
    packages = _read_package_manifest(project)
    vpm_locked = _read_vpm_manifest(project)
    risks = list(profile["risks"])
    warnings: list[str] = []
    if profile["unity_project"] and not profile["vrchat_unity_version_match"]:
        warnings.append(
            f"Unity version is {profile['unity_version'] or 'unknown'}, expected {SUPPORTED_VRCHAT_UNITY_VERSION}"
        )
    if not any(pkg in packages or pkg in vpm_locked for pkg in ("com.vrchat.avatars", "com.vrchat.worlds")):
        warnings.append("VRChat SDK Avatars or Worlds package not detected")
    if (project / "Assets" / "VRCSDK").exists():
        risks.append("legacy Assets/VRCSDK folder detected")
    editor_log = classify_editor_log(project)
    return {
        "ok": True,
        "scanned_at": _now_utc(),
        "project": profile,
        "warnings": warnings,
        "risks": risks,
        "editor_log": editor_log,
        "blocked_actions": [
            "sdk_upload",
            "package_import",
            "destructive_project_mutation",
        ],
        "dry_run_required": True,
    }


def _editor_log_candidates(project: Path) -> list[Path]:
    local = os.environ.get("LOCALAPPDATA")
    candidates = [project / "Library" / "Editor.log"]
    if local:
        candidates.append(Path(local) / "Unity" / "Editor" / "Editor.log")
    return candidates


def classify_editor_log(project: Path, max_lines: int = 400) -> dict[str, Any]:
    log_path = next((path for path in _editor_log_candidates(project) if path.is_file()), None)
    if not log_path:
        return {"available": False, "path": "", "findings": []}
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    except OSError:
        return {"available": False, "path": str(log_path), "findings": []}
    findings = []
    for line in lines:
        for rule_id, severity, pattern in EDITOR_LOG_RULES:
            if pattern.search(line):
                findings.append({"id": rule_id, "severity": severity, "line": line[:500]})
                break
    return {"available": True, "path": str(log_path), "findings": findings[:100]}


def _session_path(project: Path) -> Path:
    return project / SESSION_RELATIVE_PATH


def read_bridge_session(project_path: str) -> BridgeSession:
    project = Path(project_path).expanduser()
    data = _read_json(_session_path(project))
    token = str(data.get("token") or "")
    port = int(data.get("port") or 0)
    project_hash = str(data.get("projectHash") or data.get("project_hash") or "")
    if not token or port <= 0:
        raise UnityBridgeError(f"Unity bridge session is missing or incomplete at {_session_path(project)}")
    if project_hash and project_hash != _project_hash(project):
        raise UnityBridgeError("Unity bridge session project hash does not match project path")
    return BridgeSession(project_path=project, project_hash=project_hash, port=port, token=token)


def bridge_request(
    project_path: str,
    endpoint: str,
    timeout_seconds: float = 5.0,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session = read_bridge_session(project_path)
    if not endpoint.startswith("/"):
        raise UnityBridgeError("Bridge endpoint must start with /")
    body = None
    headers = {
        "X-Hermes-Bridge-Token": session.token,
        "User-Agent": "HermesUnityVrchatBridge/0.1.0",
    }
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    url = session.base_url + endpoint
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(1024 * 1024)
    except urllib.error.HTTPError as exc:
        raise UnityBridgeError(f"Unity bridge returned HTTP {exc.code}") from exc
    except OSError as exc:
        raise UnityBridgeError(f"Unity bridge request failed: {exc}") from exc
    try:
        return json.loads(body.decode("utf-8"))
    except ValueError as exc:
        raise UnityBridgeError("Unity bridge returned invalid JSON") from exc


def status() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    unity_package = package_root / "unity_package" / "Packages" / "com.hermes.unity-vrchat-bridge"
    return {
        "ok": True,
        "plugin": PLUGIN_ID,
        "scanned_at": _now_utc(),
        "hermes_home": str(get_hermes_home()),
        "default_port": DEFAULT_PORT,
        "supported_vrchat_unity_version": SUPPORTED_VRCHAT_UNITY_VERSION,
        "unity_editor_package_present": unity_package.is_dir(),
        "unity_editor_package_path": str(unity_package),
        "mvp_policy": {
            "read_only_first": True,
            "sdk_upload_blocked": True,
            "package_import_blocked": True,
            "dangerous_mutations_require_dry_run": True,
        },
    }


def _archive_entries(archive_path: Path) -> tuple[str, list[str]]:
    suffix = archive_path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            return "zip", archive.namelist()[:MAX_ARCHIVE_ENTRIES]
    if suffix == ".unitypackage":
        with tarfile.open(archive_path) as archive:
            return "unitypackage", archive.getnames()[:MAX_ARCHIVE_ENTRIES]
    raise ValueError("archive_path must be a .zip or .unitypackage file")


def _entry_kinds(entries: list[str]) -> dict[str, int]:
    patterns = {
        "prefab": "*.prefab",
        "scene": "*.unity",
        "fbx": "*.fbx",
        "material": "*.mat",
        "controller": "*.controller",
        "animation": "*.anim",
        "shader": "*.shader",
        "texture": ("*.png", "*.jpg", "*.jpeg", "*.tga", "*.psd"),
        "meta": "*.meta",
        "readme": ("*readme*", "*license*", "*terms*", "*利用規約*"),
    }
    counts: dict[str, int] = {}
    lower_entries = [entry.lower() for entry in entries]
    for kind, raw_patterns in patterns.items():
        pats = raw_patterns if isinstance(raw_patterns, tuple) else (raw_patterns,)
        counts[kind] = sum(
            1 for entry in lower_entries if any(fnmatch.fnmatch(entry, pattern.lower()) for pattern in pats)
        )
    return counts


def _infer_dependencies(entries: list[str], installed: dict[str, str]) -> list[dict[str, Any]]:
    haystack = "\n".join(entries).lower()
    deps: dict[str, dict[str, Any]] = {}
    for needle, (name, package_id) in DEPENDENCY_HINTS.items():
        if needle in haystack:
            deps[name] = {
                "name": name,
                "package_id": package_id,
                "status": "installed" if package_id in installed else "missing_or_unknown",
                "evidence": [f"archive path references {needle}"],
            }
    return list(deps.values())


def inspect_commercial_asset_archive(archive_path: str, project_path: str | None = None) -> dict[str, Any]:
    archive = Path(archive_path).expanduser()
    if not archive.is_file():
        return {"ok": False, "error": f"archive not found: {archive}"}
    try:
        archive_type, entries = _archive_entries(archive)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        return {"ok": False, "error": str(exc), "archive_path": str(archive)}
    installed = _read_package_manifest(Path(project_path).expanduser()) if project_path else {}
    kind_counts = _entry_kinds(entries)
    root_candidates = sorted({entry.split("/")[0] for entry in entries if "/" in entry})[:20]
    risks = []
    if kind_counts.get("readme", 0) == 0:
        risks.append("README/license/terms file not obvious from archive paths")
    if archive_type == "unitypackage" and kind_counts.get("meta", 0) == 0:
        risks.append("unitypackage contains no obvious .meta entries")
    return {
        "ok": True,
        "scanned_at": _now_utc(),
        "archive_path": str(archive),
        "archive_type": archive_type,
        "entry_count": len(entries),
        "truncated": len(entries) >= MAX_ARCHIVE_ENTRIES,
        "root_candidates": root_candidates,
        "kinds": kind_counts,
        "dependencies": _infer_dependencies(entries, installed),
        "risks": risks,
        "will_modify_files": False,
        "blocked_actions": ["redistribution", "drm_bypass", "license_bypass", "automatic_import"],
    }


def _error_response(exc: Exception) -> str:
    return _json({"ok": False, "error": str(exc)})


def handle_status(args: dict[str, Any] | None = None, **_: Any) -> str:
    return _json(status())


def handle_health(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    try:
        return _json(bridge_request(str(args.get("project_path") or ""), "/health", float(args.get("timeout_seconds") or 5)))
    except Exception as exc:
        return _error_response(exc)


def handle_snapshot(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    try:
        return _json(bridge_request(str(args.get("project_path") or ""), "/snapshot", float(args.get("timeout_seconds") or 5)))
    except Exception as exc:
        return _error_response(exc)


def handle_selection(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    try:
        return _json(bridge_request(str(args.get("project_path") or ""), "/selection", float(args.get("timeout_seconds") or 5)))
    except Exception as exc:
        return _error_response(exc)


def handle_capabilities(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    try:
        return _json(
            bridge_request(
                str(args.get("project_path") or ""),
                "/editor/capabilities",
                float(args.get("timeout_seconds") or 5),
            )
        )
    except Exception as exc:
        return _error_response(exc)


def handle_packages(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    try:
        return _json(
            bridge_request(
                str(args.get("project_path") or ""),
                "/project/packages",
                float(args.get("timeout_seconds") or 5),
            )
        )
    except Exception as exc:
        return _error_response(exc)


def handle_hierarchy(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    limit = max(1, min(int(args.get("limit") or 200), 500))
    try:
        return _json(
            bridge_request(
                str(args.get("project_path") or ""),
                f"/scene/hierarchy?limit={limit}",
                float(args.get("timeout_seconds") or 5),
            )
        )
    except Exception as exc:
        return _error_response(exc)


def handle_console_recent(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    limit = max(1, min(int(args.get("limit") or 100), 500))
    try:
        return _json(
            bridge_request(
                str(args.get("project_path") or ""),
                f"/logs/recent?limit={limit}",
                float(args.get("timeout_seconds") or 5),
            )
        )
    except Exception as exc:
        return _error_response(exc)


def handle_asset_search(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    folders = args.get("folders")
    if not isinstance(folders, list):
        folders = ["Assets"]
    limit = max(1, min(int(args.get("limit") or 100), 500))
    try:
        return _json(
            bridge_request(
                str(args.get("project_path") or ""),
                "/assets/search",
                float(args.get("timeout_seconds") or 5),
                method="POST",
                payload={
                    "filter": str(args.get("filter") or ""),
                    "folders": [str(folder) for folder in folders],
                    "limit": limit,
                },
            )
        )
    except Exception as exc:
        return _error_response(exc)


def handle_asset_info(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    paths = args.get("paths")
    if not isinstance(paths, list):
        paths = []
    try:
        return _json(
            bridge_request(
                str(args.get("project_path") or ""),
                "/assets/info",
                float(args.get("timeout_seconds") or 5),
                method="POST",
                payload={
                    "paths": [str(path) for path in paths],
                    "includeDependencies": bool(args.get("include_dependencies", False)),
                },
            )
        )
    except Exception as exc:
        return _error_response(exc)


def handle_menu_execute(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    dry_run = bool(args.get("dry_run", True))
    if not dry_run:
        return _json(
            {
                "ok": False,
                "error": "unity_bridge_menu_execute requires dry_run=true in the MVP",
                "blocked_action": "menu_execute_live",
            }
        )
    try:
        return _json(
            bridge_request(
                str(args.get("project_path") or ""),
                "/menu/execute",
                float(args.get("timeout_seconds") or 5),
                method="POST",
                payload={"menuPath": str(args.get("menu_path") or ""), "dryRun": True},
            )
        )
    except Exception as exc:
        return _error_response(exc)


def handle_operation_plan(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    dry_run = bool(args.get("dry_run", True))
    if not dry_run:
        return _json(
            {
                "ok": False,
                "error": "unity_bridge_operation_plan requires dry_run=true in the MVP",
                "blocked_action": "operation_execute_live",
            }
        )
    try:
        return _json(
            bridge_request(
                str(args.get("project_path") or ""),
                "/operation/plan",
                float(args.get("timeout_seconds") or 5),
                method="POST",
                payload={
                    "operation": str(args.get("operation") or ""),
                    "targetPath": str(args.get("target_path") or ""),
                    "dryRun": True,
                },
            )
        )
    except Exception as exc:
        return _error_response(exc)


def handle_plan_apply(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    dry_run = bool(args.get("dry_run", True))
    if not dry_run:
        return _json(
            {
                "ok": False,
                "error": "unity_bridge_plan_apply requires dry_run=true in the MVP",
                "blocked_action": "plan_apply_live",
            }
        )
    try:
        return _json(
            bridge_request(
                str(args.get("project_path") or ""),
                "/plan/apply",
                float(args.get("timeout_seconds") or 5),
                method="POST",
                payload={"operation": str(args.get("operation") or ""), "dryRun": True},
            )
        )
    except Exception as exc:
        return _error_response(exc)


def handle_project_profile(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(project_profile(str(args.get("project_path") or "")))


def handle_vrchat_project_health(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(vrchat_project_health(str(args.get("project_path") or "")))


def handle_commercial_asset_inspect(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(
        inspect_commercial_asset_archive(
            str(args.get("archive_path") or ""),
            str(args.get("project_path") or "") or None,
        )
    )


def handle_slash(command: str = "", **_: Any) -> str:
    argv = command.split()
    if argv and argv[0].lstrip("/") == "unity-vrchat-bridge":
        argv = argv[1:]
    return run_cli(argv)


def run_cli(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(prog="hermes unity-vrchat-bridge")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status")
    for name in (
        "health",
        "snapshot",
        "selection",
        "capabilities",
        "packages",
        "hierarchy",
        "console",
        "project-profile",
        "vrchat-health",
    ):
        p = sub.add_parser(name)
        p.add_argument("project_path")
        if name == "hierarchy":
            p.add_argument("--limit", type=int, default=200)
    asset = sub.add_parser("asset-search")
    asset.add_argument("project_path")
    asset.add_argument("filter")
    asset.add_argument("--folder", action="append", default=[])
    asset.add_argument("--limit", type=int, default=100)
    info = sub.add_parser("asset-info")
    info.add_argument("project_path")
    info.add_argument("paths", nargs="+")
    info.add_argument("--include-dependencies", action="store_true")
    menu = sub.add_parser("menu-execute")
    menu.add_argument("project_path")
    menu.add_argument("menu_path")
    menu.add_argument("--live", action="store_true")
    op = sub.add_parser("operation-plan")
    op.add_argument("project_path")
    op.add_argument("operation")
    op.add_argument("--target-path", default="")
    op.add_argument("--live", action="store_true")
    plan = sub.add_parser("plan-apply")
    plan.add_argument("project_path")
    plan.add_argument("operation")
    plan.add_argument("--live", action="store_true")
    inspect = sub.add_parser("inspect-archive")
    inspect.add_argument("archive_path")
    inspect.add_argument("--project-path", default="")
    ns = parser.parse_args(argv or ["status"])
    if ns.command in (None, "status"):
        return _json(status())
    if ns.command == "health":
        return handle_health({"project_path": ns.project_path})
    if ns.command == "snapshot":
        return handle_snapshot({"project_path": ns.project_path})
    if ns.command == "selection":
        return handle_selection({"project_path": ns.project_path})
    if ns.command == "capabilities":
        return handle_capabilities({"project_path": ns.project_path})
    if ns.command == "packages":
        return handle_packages({"project_path": ns.project_path})
    if ns.command == "hierarchy":
        return handle_hierarchy({"project_path": ns.project_path, "limit": ns.limit})
    if ns.command == "console":
        return handle_console_recent({"project_path": ns.project_path})
    if ns.command == "asset-search":
        return handle_asset_search(
            {
                "project_path": ns.project_path,
                "filter": ns.filter,
                "folders": ns.folder or ["Assets"],
                "limit": ns.limit,
            }
        )
    if ns.command == "asset-info":
        return handle_asset_info(
            {
                "project_path": ns.project_path,
                "paths": ns.paths,
                "include_dependencies": ns.include_dependencies,
            }
        )
    if ns.command == "menu-execute":
        return handle_menu_execute(
            {
                "project_path": ns.project_path,
                "menu_path": ns.menu_path,
                "dry_run": not ns.live,
            }
        )
    if ns.command == "operation-plan":
        return handle_operation_plan(
            {
                "project_path": ns.project_path,
                "operation": ns.operation,
                "target_path": ns.target_path,
                "dry_run": not ns.live,
            }
        )
    if ns.command == "plan-apply":
        return handle_plan_apply(
            {
                "project_path": ns.project_path,
                "operation": ns.operation,
                "dry_run": not ns.live,
            }
        )
    if ns.command == "project-profile":
        return handle_project_profile({"project_path": ns.project_path})
    if ns.command == "vrchat-health":
        return handle_vrchat_project_health({"project_path": ns.project_path})
    if ns.command == "inspect-archive":
        return handle_commercial_asset_inspect(
            {"archive_path": ns.archive_path, "project_path": ns.project_path}
        )
    return _json({"ok": False, "error": f"unknown command: {ns.command}"})

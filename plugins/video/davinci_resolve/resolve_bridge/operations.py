"""Small, defensive wrapper around the DaVinci Resolve scripting API."""

from __future__ import annotations

import importlib
import csv
import os
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {
    ".3g2", ".3gp", ".avi", ".braw", ".cin", ".crm", ".dv", ".flv", ".m2ts",
    ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".mxf", ".r3d", ".ts",
    ".vob", ".webm",
}

AUDIO_EXTENSIONS = {
    ".aac", ".aif", ".aiff", ".bwf", ".flac", ".m4a", ".mp3", ".ogg", ".wav",
    ".wma",
}

IMAGE_EXTENSIONS = {
    ".ari", ".arw", ".bmp", ".cr2", ".cr3", ".dng", ".dpx", ".exr", ".gif",
    ".heic", ".jpeg", ".jpg", ".nef", ".png", ".psd", ".raf", ".tif", ".tiff",
}

DEFAULT_MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS | IMAGE_EXTENSIONS

COMMON_MODULE_PATHS = [
    Path("/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules"),
    Path("/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion"),
]

COMMON_APP_PATHS = [
    Path("/Applications/DaVinci Resolve/DaVinci Resolve.app"),
    Path("/Applications/DaVinci Resolve/DaVinci Resolve Studio.app"),
    Path("/Applications/DaVinci Resolve 20/DaVinci Resolve.app"),
    Path("/Applications/DaVinci Resolve Studio 20/DaVinci Resolve Studio.app"),
    Path("/Applications/DaVinci Resolve 21 Beta/DaVinci Resolve.app"),
    Path("/Applications/DaVinci Resolve Studio 21 Beta/DaVinci Resolve Studio.app"),
    Path("/Applications/DaVinci Resolve 21 Beta 3/DaVinci Resolve.app"),
    Path("/Applications/DaVinci Resolve Studio 21 Beta 3/DaVinci Resolve Studio.app"),
    Path("/Applications/DaVinci Resolve.app"),
    Path("/Applications/DaVinci Resolve Studio.app"),
    Path("/Applications/DaVinci Resolve 20.app"),
    Path("/Applications/DaVinci Resolve Studio 20.app"),
    Path("/Applications/DaVinci Resolve 21 Beta.app"),
    Path("/Applications/DaVinci Resolve Studio 21 Beta.app"),
    Path("/Applications/DaVinci Resolve 21 Beta 3.app"),
    Path("/Applications/DaVinci Resolve Studio 21 Beta 3.app"),
]

RESOLVE_APP_PATHS = [
    Path("/Applications/DaVinci Resolve/DaVinci Resolve.app"),
    Path("/Applications/DaVinci Resolve.app"),
]

STUDIO_APP_PATHS = [
    Path("/Applications/DaVinci Resolve/DaVinci Resolve Studio.app"),
    Path("/Applications/DaVinci Resolve Studio.app"),
]

RESOLVE_20_APP_PATHS = [
    Path("/Applications/DaVinci Resolve 20/DaVinci Resolve.app"),
    Path("/Applications/DaVinci Resolve 20.app"),
    Path("/Applications/DaVinci Resolve/DaVinci Resolve.app"),
    Path("/Applications/DaVinci Resolve.app"),
]

STUDIO_20_APP_PATHS = [
    Path("/Applications/DaVinci Resolve Studio 20/DaVinci Resolve Studio.app"),
    Path("/Applications/DaVinci Resolve Studio 20.app"),
    Path("/Applications/DaVinci Resolve/DaVinci Resolve Studio.app"),
    Path("/Applications/DaVinci Resolve Studio.app"),
]

RESOLVE_21_APP_PATHS = [
    Path("/Applications/DaVinci Resolve 21 Beta 3/DaVinci Resolve.app"),
    Path("/Applications/DaVinci Resolve 21 Beta/DaVinci Resolve.app"),
    Path("/Applications/DaVinci Resolve 21 Beta 3.app"),
    Path("/Applications/DaVinci Resolve 21 Beta.app"),
]

STUDIO_21_APP_PATHS = [
    Path("/Applications/DaVinci Resolve Studio 21 Beta 3/DaVinci Resolve Studio.app"),
    Path("/Applications/DaVinci Resolve Studio 21 Beta/DaVinci Resolve Studio.app"),
    Path("/Applications/DaVinci Resolve Studio 21 Beta 3.app"),
    Path("/Applications/DaVinci Resolve Studio 21 Beta.app"),
]


def _ok(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def _error(message: str, **payload: Any) -> dict[str, Any]:
    return {"ok": False, "error": message, **payload}


def _add_existing_module_paths() -> list[str]:
    added: list[str] = []
    env_api = os.environ.get("RESOLVE_SCRIPT_API")
    if env_api:
        api_modules = Path(env_api).expanduser() / "Modules"
        if api_modules.exists():
            sys.path.insert(0, str(api_modules))
            added.append(str(api_modules))

    for path in COMMON_MODULE_PATHS:
        if path.exists():
            sys.path.insert(0, str(path))
            added.append(str(path))

    return added


def _import_resolve_module() -> tuple[Any | None, str | None, list[str]]:
    added_paths = _add_existing_module_paths()
    try:
        return importlib.import_module("DaVinciResolveScript"), None, added_paths
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", added_paths


def _connect() -> tuple[Any | None, dict[str, Any] | None]:
    module, import_error, added_paths = _import_resolve_module()
    if module is None:
        return None, _error(
            "DaVinciResolveScript could not be imported.",
            import_error=import_error,
            added_sys_paths=added_paths,
        )

    try:
        resolve = module.scriptapp("Resolve")
    except Exception as exc:
        return None, _error(
            "DaVinci Resolve scripting module imported, but scriptapp failed.",
            resolve_error=f"{type(exc).__name__}: {exc}",
            added_sys_paths=added_paths,
        )

    if resolve is None:
        return None, _error(
            "DaVinci Resolve is not reachable. Open Resolve and enable scripting support.",
            added_sys_paths=added_paths,
        )

    return resolve, None


def _current_project(resolve: Any) -> tuple[Any | None, dict[str, Any] | None]:
    project_manager = resolve.GetProjectManager()
    if not project_manager:
        return None, _error("Resolve project manager is unavailable.")

    project = project_manager.GetCurrentProject()
    if not project:
        return None, _error("No active DaVinci Resolve project is open.")

    return project, None


def _media_pool(project: Any) -> tuple[Any | None, dict[str, Any] | None]:
    media_pool = project.GetMediaPool()
    if not media_pool:
        return None, _error("Current Resolve project has no accessible media pool.")
    return media_pool, None


def _guard_mutation(dry_run: bool, confirm: bool, action: str) -> dict[str, Any] | None:
    if dry_run:
        return None
    if not confirm:
        return _error(
            f"{action} requires confirm=true when dry_run=false.",
            dry_run=dry_run,
            confirm=confirm,
        )
    return None


def _expand_paths(paths: list[str]) -> list[str]:
    return [str(Path(path).expanduser().resolve()) for path in paths]


def _path_report(paths: list[str]) -> dict[str, list[str]]:
    existing = [path for path in paths if Path(path).exists()]
    missing = [path for path in paths if not Path(path).exists()]
    return {"existing_paths": existing, "missing_paths": missing}


def _default_output_dir() -> Path:
    candidates = [
        Path.home() / "Documents" / "Hermes Resolve Exports",
        Path(tempfile.gettempdir()) / "hermes-resolve-exports",
    ]
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            continue
    raise OSError("Could not create a writable Resolve interchange output directory.")


def _safe_file_stem(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return safe.strip("_") or "hermes_resolve"


def _output_path(output_path: str | None, name: str, suffix: str) -> Path:
    if output_path:
        path = Path(output_path).expanduser().resolve()
    else:
        path = _default_output_dir() / f"{_safe_file_stem(name)}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _file_uri(path: str) -> str:
    resolved = Path(path).expanduser().resolve()
    return resolved.as_uri()


def _seconds_to_frames(seconds: float, frame_rate: int) -> int:
    return max(1, int(round(seconds * frame_rate)))


def _fcpx_time(frames: int, frame_rate: int) -> str:
    return f"{int(frames)}/{int(frame_rate)}s"


def _media_kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    return "unknown"


def _media_type_value(media_type: str | int | None) -> int | None:
    if media_type is None:
        return None
    if isinstance(media_type, int):
        return media_type
    normalized = media_type.lower().strip()
    if normalized == "video":
        return 1
    if normalized == "audio":
        return 2
    return None


def _duration_from_ffprobe(path: str) -> float | None:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def _clip_plan_paths(clips: list[dict[str, Any]], music_paths: list[str] | None = None) -> list[str]:
    paths = [str(clip["path"]) for clip in clips if clip.get("path")]
    paths.extend(music_paths or [])
    return paths


def _make_path_item_map(imported_items: list[Any], imported_paths: list[str]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for path, item in zip(imported_paths, imported_items):
        if item:
            mapping[path] = item
    return mapping


def _add_markers_to_timeline(timeline: Any, markers: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    added: list[dict[str, Any]] = []
    for marker in markers or []:
        frame = int(marker.get("frame", 0))
        color = str(marker.get("color", "Blue"))
        name = str(marker.get("name", "Marker"))
        note = str(marker.get("note", ""))
        duration = int(marker.get("duration", 1))
        ok = timeline.AddMarker(frame, color, name, note, duration, str(marker.get("custom_data", "")))
        added.append({"frame": frame, "name": name, "added": bool(ok)})
    return added


def _probe_recommendation(
    module_imported: bool,
    resolve_reachable: bool,
    existing_app_paths: list[str],
) -> dict[str, Any]:
    studio_installed = any("Studio" in path for path in existing_app_paths)
    resolve_installed = any("DaVinci Resolve.app" in path and "Studio" not in path for path in existing_app_paths)
    if resolve_reachable:
        return {
            "recommended_mode": "studio_live_control",
            "live_control_available": True,
            "free_interchange_available": True,
            "likely_reason": None,
        }
    if module_imported and resolve_installed and not studio_installed:
        return {
            "recommended_mode": "free_interchange",
            "live_control_available": False,
            "free_interchange_available": True,
            "likely_reason": "free_resolve_external_scripting_unavailable",
            "next_steps": [
                "Use resolve_generate_fcpxml_timeline or resolve_generate_marker_csv for free Resolve workflows.",
                "Install DaVinci Resolve Studio and enable Preferences > System > General > External scripting using: Local for live control.",
            ],
        }
    return {
        "recommended_mode": "diagnostic",
        "live_control_available": False,
        "free_interchange_available": True,
        "likely_reason": "resolve_scripting_not_reachable",
        "next_steps": [
            "Confirm Resolve is fully open.",
            "If using Studio, enable Preferences > System > General > External scripting using: Local.",
            "If using free Resolve, use interchange tools instead of live-control tools.",
        ],
    }


def probe() -> dict[str, Any]:
    module, import_error, added_paths = _import_resolve_module()
    resolve_reachable = False
    resolve_error = None
    current_project = None

    if module is not None:
        try:
            resolve = module.scriptapp("Resolve")
            resolve_reachable = resolve is not None
            if resolve_reachable:
                project_manager = resolve.GetProjectManager()
                project = project_manager.GetCurrentProject() if project_manager else None
                current_project = project.GetName() if project else None
        except Exception as exc:
            resolve_error = f"{type(exc).__name__}: {exc}"

    existing_app_paths = [str(path) for path in COMMON_APP_PATHS if path.exists()]
    recommendation = _probe_recommendation(
        module_imported=module is not None,
        resolve_reachable=resolve_reachable,
        existing_app_paths=existing_app_paths,
    )

    return _ok(
        platform=platform.platform(),
        python=sys.version.split()[0],
        environment={
            "RESOLVE_SCRIPT_API": os.environ.get("RESOLVE_SCRIPT_API"),
            "RESOLVE_SCRIPT_LIB": os.environ.get("RESOLVE_SCRIPT_LIB"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
        existing_app_paths=existing_app_paths,
        existing_module_paths=[str(path) for path in COMMON_MODULE_PATHS if path.exists()],
        added_sys_paths=added_paths,
        module_imported=module is not None,
        import_error=import_error,
        resolve_reachable=resolve_reachable,
        resolve_error=resolve_error,
        current_project=current_project,
        **recommendation,
    )


def capabilities() -> dict[str, Any]:
    return _ok(
        purpose=(
            "Control a local DaVinci Resolve session through Resolve's Python scripting API."
        ),
        recommended_workflow=[
            "Call resolve_capabilities when unsure what this plugin can do.",
            "Call resolve_launch to open Resolve on the local Mac.",
            "Call resolve_probe to choose studio_live_control or free_interchange mode.",
            "In Studio mode, call resolve_project_summary to inspect the current project and timeline.",
            "For script-driven edits, scan media folders, create a structured edit plan, dry-run resolve_create_scripted_timeline, then render after approval.",
            "For edits, call the mutating tool with dry_run=true first.",
            "After user approval, call the same mutating tool with dry_run=false and confirm=true.",
            "Call resolve_project_summary again to verify the result.",
            "For free Resolve, generate FCPXML/CSV interchange files and ask the user to import them.",
        ],
        tools={
            "resolve_launch": "Open DaVinci Resolve or Resolve Studio locally.",
            "resolve_probe": "Check whether Resolve scripting is installed and reachable.",
            "resolve_project_summary": "Read the active project and timeline state.",
            "resolve_import_media": "Import footage, audio, stills, or other media into the media pool.",
            "resolve_create_timeline": "Create an empty timeline or one seeded with media.",
            "resolve_append_to_current_timeline": "Import media if needed and append it to the active timeline.",
            "resolve_add_timeline_marker": "Add a marker to the active timeline at a frame.",
            "resolve_scan_media_folder": "List usable video, audio, and image media in a local folder for edit planning.",
            "resolve_create_scripted_timeline": "Build a Studio timeline from a structured edit plan with clip ranges, tracks, music, and markers.",
            "resolve_render_timeline": "Configure and start a Resolve Studio render job for the current timeline.",
            "resolve_render_status": "Check render progress for a Resolve render job.",
            "resolve_generate_fcpxml_timeline": "Create an FCPXML timeline file that free Resolve can import manually.",
            "resolve_generate_marker_csv": "Create a marker CSV/manifest for free Resolve workflows.",
        },
        modes={
            "studio_live_control": [
                "Requires DaVinci Resolve Studio.",
                "Requires Preferences > System > General > External scripting using: Local.",
                "Supports live project inspection and project mutation through Resolve's scripting API.",
            ],
            "free_interchange": [
                "Works with free DaVinci Resolve because it produces importable files instead of using external scripting.",
                "Requires the user or agent UI automation to import the generated FCPXML/CSV in Resolve.",
                "Does not provide live project inspection.",
            ],
        },
        current_boundaries=[
            "The plugin controls Resolve through the scripting API, not by clicking the UI.",
            "Resolve must be installed locally and usually must finish opening before control tools work.",
            "Free DaVinci Resolve does not expose reliable external live control; use interchange tools there.",
            "If Resolve 20 and Resolve 21 beta are installed side by side, use resolve_launch with app_path to pick the exact app bundle.",
            "Do not open production projects in Resolve 21 beta without a full project library backup and individual project backups.",
            "Complex color, Fusion, Fairlight, and UI-only operations may need more tool coverage.",
            "Opening a specific project by database name and finer timeline placement are planned extensions.",
        ],
        safety_rules=[
            "Never delete media, timelines, bins, grades, or render jobs unless the user explicitly asks.",
            "Use dry_run=true before import, append, timeline creation, or marker changes.",
            "Use confirm=true only after the user approves the dry-run plan.",
            "Prefer read-only inspection when project state is unknown.",
        ],
        example_prompts=[
            "Open Resolve, inspect the current project, then tell me what timelines exist.",
            "Dry-run importing this folder of music into a bin named Score.",
            "Add a blue timeline marker at frame 0 named Hermes smoke test.",
            "Create a new timeline from these clips after showing me the dry-run plan.",
            "Scan this footage folder, build a 60-second cut from my script, add this music, and render a QuickTime.",
            "Generate an FCPXML timeline I can import into free Resolve from these clips.",
        ],
    )


def scan_media_folder(
    folder_path: str,
    recursive: bool = True,
    include_extensions: list[str] | None = None,
    include_durations: bool = False,
    max_files: int = 500,
) -> dict[str, Any]:
    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists():
        return _error("Folder does not exist.", folder_path=str(folder))
    if not folder.is_dir():
        return _error("folder_path must be a directory.", folder_path=str(folder))
    if max_files < 1 or max_files > 5000:
        return _error("max_files must be between 1 and 5000.", max_files=max_files)

    extensions = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in (include_extensions or sorted(DEFAULT_MEDIA_EXTENSIONS))
    }
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    files: list[dict[str, Any]] = []
    skipped_count = 0
    for path in sorted(iterator):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if len(files) >= max_files:
            skipped_count += 1
            continue
        payload: dict[str, Any] = {
            "path": str(path),
            "name": path.name,
            "stem": path.stem,
            "extension": path.suffix.lower(),
            "kind": _media_kind(str(path)),
            "size_bytes": path.stat().st_size,
        }
        if include_durations and payload["kind"] in {"video", "audio"}:
            payload["duration_seconds"] = _duration_from_ffprobe(str(path))
        files.append(payload)

    counts = {"video": 0, "audio": 0, "image": 0, "unknown": 0}
    for item in files:
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1

    return _ok(
        folder_path=str(folder),
        recursive=recursive,
        include_extensions=sorted(extensions),
        include_durations=include_durations,
        max_files=max_files,
        returned_count=len(files),
        skipped_count=skipped_count,
        counts=counts,
        files=files,
    )


def generate_fcpxml_timeline(
    name: str,
    media_paths: list[str],
    output_path: str | None = None,
    frame_rate: int = 24,
    clip_duration_seconds: float = 5.0,
    width: int = 1920,
    height: int = 1080,
    markers: list[dict[str, Any]] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if frame_rate <= 0 or frame_rate > 240:
        return _error("frame_rate must be between 1 and 240.", frame_rate=frame_rate)
    if clip_duration_seconds <= 0:
        return _error("clip_duration_seconds must be greater than 0.", clip_duration_seconds=clip_duration_seconds)
    if width <= 0 or height <= 0:
        return _error("width and height must be positive.", width=width, height=height)

    expanded_paths = _expand_paths(media_paths)
    report = _path_report(expanded_paths)
    destination = _output_path(output_path, name, ".fcpxml")
    clip_duration_frames = _seconds_to_frames(clip_duration_seconds, frame_rate)
    total_duration_frames = clip_duration_frames * len(expanded_paths)
    plan = {
        "action": "generate_fcpxml_timeline",
        "mode": "free_interchange",
        "timeline_name": name,
        "output_path": str(destination),
        "media_paths": expanded_paths,
        "frame_rate": frame_rate,
        "clip_duration_seconds": clip_duration_seconds,
        "width": width,
        "height": height,
        "markers": markers or [],
        **report,
    }
    if dry_run:
        return _ok(dry_run=True, plan=plan)
    if report["missing_paths"]:
        return _error("Cannot generate FCPXML with missing media files.", **plan)

    fcpxml = ET.Element("fcpxml", {"version": "1.10"})
    resources = ET.SubElement(fcpxml, "resources")
    ET.SubElement(
        resources,
        "format",
        {
            "id": "r1",
            "name": f"FFVideoFormat{height}p{frame_rate}",
            "frameDuration": _fcpx_time(1, frame_rate),
            "width": str(width),
            "height": str(height),
            "colorSpace": "1-1-1 (Rec. 709)",
        },
    )
    for index, media_path in enumerate(expanded_paths, start=2):
        media_file = Path(media_path)
        ET.SubElement(
            resources,
            "asset",
            {
                "id": f"r{index}",
                "name": media_file.stem,
                "src": _file_uri(media_path),
                "start": "0s",
                "duration": _fcpx_time(clip_duration_frames, frame_rate),
                "hasVideo": "1",
                "hasAudio": "1",
                "audioSources": "1",
                "audioChannels": "2",
                "audioRate": "48000",
            },
        )

    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", {"name": "Hermes Interchange"})
    project = ET.SubElement(event, "project", {"name": name})
    sequence = ET.SubElement(
        project,
        "sequence",
        {
            "format": "r1",
            "duration": _fcpx_time(total_duration_frames, frame_rate),
            "tcStart": "0s",
            "tcFormat": "NDF",
        },
    )
    spine = ET.SubElement(sequence, "spine")
    normalized_markers = markers or []
    for index, media_path in enumerate(expanded_paths, start=2):
        offset_frames = (index - 2) * clip_duration_frames
        asset_clip = ET.SubElement(
            spine,
            "asset-clip",
            {
                "name": Path(media_path).stem,
                "ref": f"r{index}",
                "offset": _fcpx_time(offset_frames, frame_rate),
                "start": "0s",
                "duration": _fcpx_time(clip_duration_frames, frame_rate),
            },
        )
        for marker in normalized_markers:
            marker_frame = int(marker.get("frame", 0))
            if offset_frames <= marker_frame < offset_frames + clip_duration_frames:
                ET.SubElement(
                    asset_clip,
                    "marker",
                    {
                        "start": _fcpx_time(marker_frame - offset_frames, frame_rate),
                        "value": str(marker.get("name", "Marker")),
                        "note": str(marker.get("note", "")),
                    },
                )

    tree = ET.ElementTree(fcpxml)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    text = destination.read_text(encoding="utf-8")
    destination.write_text(text.replace("?>\n", "?>\n<!DOCTYPE fcpxml>\n", 1), encoding="utf-8")

    return _ok(
        dry_run=False,
        artifact_type="fcpxml",
        import_instructions=[
            "Open free DaVinci Resolve.",
            "Use File > Import > Timeline > Import AAF, EDL, XML...",
            f"Select {destination}.",
        ],
        **plan,
    )


def generate_marker_csv(
    markers: list[dict[str, Any]],
    output_path: str | None = None,
    name: str = "Hermes Markers",
    frame_rate: int = 24,
    dry_run: bool = False,
) -> dict[str, Any]:
    destination = _output_path(output_path, name, ".markers.csv")
    normalized = []
    for marker in markers:
        frame = int(marker.get("frame", 0))
        normalized.append(
            {
                "frame": max(0, frame),
                "timecode_seconds": f"{max(0, frame) / frame_rate:.3f}",
                "name": str(marker.get("name", "Marker")),
                "color": str(marker.get("color", "Blue")),
                "note": str(marker.get("note", "")),
                "duration": int(marker.get("duration", 1)),
            }
        )
    plan = {
        "action": "generate_marker_csv",
        "mode": "free_interchange",
        "output_path": str(destination),
        "name": name,
        "frame_rate": frame_rate,
        "markers": normalized,
    }
    if dry_run:
        return _ok(dry_run=True, plan=plan)

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "timecode_seconds", "name", "color", "note", "duration"],
        )
        writer.writeheader()
        writer.writerows(normalized)

    return _ok(
        dry_run=False,
        artifact_type="marker_csv",
        import_instructions=[
            "Use this CSV as a marker manifest for free Resolve workflows.",
            "For live marker insertion, use Resolve Studio or an in-Resolve helper script.",
        ],
        **plan,
    )


def _candidate_app_paths(variant: str) -> list[Path]:
    if variant == "resolve":
        return RESOLVE_APP_PATHS
    if variant == "studio":
        return STUDIO_APP_PATHS
    if variant == "beta":
        return [*STUDIO_21_APP_PATHS, *RESOLVE_21_APP_PATHS]
    if variant == "resolve20":
        return RESOLVE_20_APP_PATHS
    if variant == "studio20":
        return STUDIO_20_APP_PATHS
    if variant == "resolve21":
        return RESOLVE_21_APP_PATHS
    if variant == "studio21":
        return STUDIO_21_APP_PATHS
    return [*STUDIO_APP_PATHS, *RESOLVE_APP_PATHS]


def launch_resolve(
    variant: str = "auto",
    wait_seconds: int = 8,
    app_path: str | None = None,
) -> dict[str, Any]:
    valid_variants = {
        "auto",
        "resolve",
        "studio",
        "beta",
        "resolve20",
        "studio20",
        "resolve21",
        "studio21",
    }
    if variant not in valid_variants:
        return _error(
            "variant must be one of: auto, resolve, studio, beta, resolve20, studio20, resolve21, studio21.",
            variant=variant,
        )
    if wait_seconds < 0 or wait_seconds > 120:
        return _error("wait_seconds must be between 0 and 120.", wait_seconds=wait_seconds)

    command = None
    selected_app_path = None
    if app_path:
        explicit_path = Path(app_path).expanduser().resolve()
        if not explicit_path.exists():
            return _error("Explicit app_path does not exist.", app_path=str(explicit_path))
        if explicit_path.suffix != ".app":
            return _error("Explicit app_path must point to a macOS .app bundle.", app_path=str(explicit_path))
        selected_app_path = str(explicit_path)
        command = ["open", selected_app_path]
    else:
        for candidate in _candidate_app_paths(variant):
            if candidate.exists():
                selected_app_path = str(candidate)
                command = ["open", selected_app_path]
                break

    if command is None:
        app_name = "DaVinci Resolve Studio" if variant in {"studio", "studio20", "studio21"} else "DaVinci Resolve"
        command = ["open", "-a", app_name]

    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except Exception as exc:
        return _error(
            "Failed to start the macOS open command.",
            launch_error=f"{type(exc).__name__}: {exc}",
            command=command,
            app_path=selected_app_path,
        )

    if completed.returncode != 0:
        return _error(
            "macOS open command did not launch Resolve.",
            command=command,
            app_path=selected_app_path,
            returncode=completed.returncode,
            stderr=completed.stderr.strip(),
            stdout=completed.stdout.strip(),
        )

    if wait_seconds:
        time.sleep(wait_seconds)

    post_launch_probe = probe()
    return _ok(
        launched=True,
        command=command,
        app_path=selected_app_path,
        variant=variant,
        wait_seconds=wait_seconds,
        post_launch_probe=post_launch_probe,
    )


def project_summary() -> dict[str, Any]:
    resolve, error = _connect()
    if error:
        return error

    project, error = _current_project(resolve)
    if error:
        return error

    timeline = project.GetCurrentTimeline()
    timeline_count = project.GetTimelineCount()
    timeline_names = []
    for index in range(1, timeline_count + 1):
        item = project.GetTimelineByIndex(index)
        if item:
            timeline_names.append(item.GetName())

    timeline_payload = None
    if timeline:
        markers = timeline.GetMarkers() or {}
        timeline_payload = {
            "name": timeline.GetName(),
            "start_frame": timeline.GetStartFrame(),
            "end_frame": timeline.GetEndFrame(),
            "setting_timeline_frame_rate": timeline.GetSetting("timelineFrameRate"),
            "marker_count": len(markers),
        }

    return _ok(
        project_name=project.GetName(),
        timeline_count=timeline_count,
        timelines=timeline_names,
        current_timeline=timeline_payload,
    )


def import_media(
    paths: list[str],
    bin_name: str | None = None,
    dry_run: bool = True,
    confirm: bool = False,
) -> dict[str, Any]:
    guard = _guard_mutation(dry_run, confirm, "Importing media")
    if guard:
        return guard

    expanded_paths = _expand_paths(paths)
    report = _path_report(expanded_paths)
    plan = {
        "action": "import_media",
        "paths": expanded_paths,
        "bin_name": bin_name,
        **report,
    }
    if dry_run:
        return _ok(dry_run=True, plan=plan)
    if report["missing_paths"]:
        return _error("Cannot import missing media files.", **plan)

    resolve, error = _connect()
    if error:
        return error

    project, error = _current_project(resolve)
    if error:
        return error

    media_pool, error = _media_pool(project)
    if error:
        return error

    if bin_name:
        root_folder = media_pool.GetRootFolder()
        target = media_pool.AddSubFolder(root_folder, bin_name)
        if target:
            media_pool.SetCurrentFolder(target)

    imported = media_pool.ImportMedia(expanded_paths) or []
    return _ok(
        dry_run=False,
        imported_count=len(imported),
        imported_names=[item.GetName() for item in imported if item],
        **plan,
    )


def create_timeline(
    name: str,
    media_paths: list[str] | None = None,
    dry_run: bool = True,
    confirm: bool = False,
) -> dict[str, Any]:
    guard = _guard_mutation(dry_run, confirm, "Creating a timeline")
    if guard:
        return guard

    expanded_paths = _expand_paths(media_paths or [])
    report = _path_report(expanded_paths)
    plan = {
        "action": "create_timeline",
        "name": name,
        "media_paths": expanded_paths,
        **report,
    }
    if dry_run:
        return _ok(dry_run=True, plan=plan)
    if report["missing_paths"]:
        return _error("Cannot create seeded timeline with missing media files.", **plan)

    resolve, error = _connect()
    if error:
        return error

    project, error = _current_project(resolve)
    if error:
        return error

    media_pool, error = _media_pool(project)
    if error:
        return error

    media_items = media_pool.ImportMedia(expanded_paths) if expanded_paths else []
    if media_items:
        timeline = media_pool.CreateTimelineFromClips(name, media_items)
    else:
        timeline = media_pool.CreateEmptyTimeline(name)

    if not timeline:
        return _error("Resolve did not create the timeline.", **plan)

    return _ok(dry_run=False, timeline_name=timeline.GetName(), **plan)


def append_to_current_timeline(
    media_paths: list[str],
    dry_run: bool = True,
    confirm: bool = False,
) -> dict[str, Any]:
    guard = _guard_mutation(dry_run, confirm, "Appending media to the current timeline")
    if guard:
        return guard

    expanded_paths = _expand_paths(media_paths)
    report = _path_report(expanded_paths)
    plan = {
        "action": "append_to_current_timeline",
        "media_paths": expanded_paths,
        **report,
    }
    if dry_run:
        return _ok(dry_run=True, plan=plan)
    if report["missing_paths"]:
        return _error("Cannot append missing media files.", **plan)

    resolve, error = _connect()
    if error:
        return error

    project, error = _current_project(resolve)
    if error:
        return error

    if not project.GetCurrentTimeline():
        return _error("No current timeline is active.", **plan)

    media_pool, error = _media_pool(project)
    if error:
        return error

    media_items = media_pool.ImportMedia(expanded_paths) or []
    appended = media_pool.AppendToTimeline(media_items) or []
    return _ok(
        dry_run=False,
        imported_count=len(media_items),
        appended_count=len(appended),
        appended_names=[item.GetName() for item in media_items if item],
        **plan,
    )


def add_timeline_marker(
    frame: int,
    name: str,
    color: str = "Blue",
    note: str = "",
    duration: int = 1,
    dry_run: bool = True,
    confirm: bool = False,
) -> dict[str, Any]:
    guard = _guard_mutation(dry_run, confirm, "Adding a timeline marker")
    if guard:
        return guard

    plan = {
        "action": "add_timeline_marker",
        "frame": frame,
        "color": color,
        "name": name,
        "note": note,
        "duration": duration,
    }
    if dry_run:
        return _ok(dry_run=True, plan=plan)

    resolve, error = _connect()
    if error:
        return error

    project, error = _current_project(resolve)
    if error:
        return error

    timeline = project.GetCurrentTimeline()
    if not timeline:
        return _error("No current timeline is active.", **plan)

    added = timeline.AddMarker(frame, color, name, note, duration, "")
    if not added:
        return _error("Resolve did not add the marker.", **plan)

    return _ok(dry_run=False, added=True, **plan)


def create_scripted_timeline(
    name: str,
    clips: list[dict[str, Any]],
    music_paths: list[str] | None = None,
    markers: list[dict[str, Any]] | None = None,
    bin_name: str | None = "Hermes Edit",
    dry_run: bool = True,
    confirm: bool = False,
) -> dict[str, Any]:
    guard = _guard_mutation(dry_run, confirm, "Creating a scripted timeline")
    if guard:
        return guard
    if not clips:
        return _error("clips must contain at least one clip plan item.")

    expanded_clip_paths = _expand_paths([str(clip["path"]) for clip in clips if clip.get("path")])
    expanded_music_paths = _expand_paths(music_paths or [])
    all_paths = [*expanded_clip_paths, *expanded_music_paths]
    report = _path_report(all_paths)
    normalized_clips: list[dict[str, Any]] = []
    for index, clip in enumerate(clips):
        path = str(Path(str(clip.get("path", ""))).expanduser().resolve())
        clip_plan: dict[str, Any] = {
            "index": index,
            "path": path,
            "name": clip.get("name") or Path(path).stem,
            "start_frame": int(clip.get("start_frame", 0)),
            "end_frame": int(clip["end_frame"]) if clip.get("end_frame") is not None else None,
            "record_frame": int(clip["record_frame"]) if clip.get("record_frame") is not None else None,
            "media_type": clip.get("media_type"),
            "track_index": int(clip["track_index"]) if clip.get("track_index") is not None else None,
            "note": clip.get("note", ""),
        }
        if clip_plan["start_frame"] < 0:
            return _error("clip start_frame cannot be negative.", clip=clip_plan)
        if clip_plan["end_frame"] is not None and clip_plan["end_frame"] <= clip_plan["start_frame"]:
            return _error("clip end_frame must be greater than start_frame.", clip=clip_plan)
        normalized_clips.append(clip_plan)

    normalized_music = [
        {
            "path": path,
            "name": Path(path).stem,
            "media_type": "audio",
            "track_index": 1,
        }
        for path in expanded_music_paths
    ]
    plan = {
        "action": "create_scripted_timeline",
        "mode": "studio_live_control",
        "name": name,
        "bin_name": bin_name,
        "clips": normalized_clips,
        "music": normalized_music,
        "markers": markers or [],
        **report,
    }
    if dry_run:
        return _ok(dry_run=True, plan=plan)
    if report["missing_paths"]:
        return _error("Cannot create scripted timeline with missing media files.", **plan)

    resolve, error = _connect()
    if error:
        return error

    project, error = _current_project(resolve)
    if error:
        return error

    media_pool, error = _media_pool(project)
    if error:
        return error

    if bin_name:
        root_folder = media_pool.GetRootFolder()
        target = media_pool.AddSubFolder(root_folder, bin_name)
        if target:
            media_pool.SetCurrentFolder(target)

    imported_items = media_pool.ImportMedia(all_paths) or []
    item_by_path = _make_path_item_map(imported_items, all_paths)
    timeline = media_pool.CreateEmptyTimeline(name)
    if not timeline:
        return _error("Resolve did not create the scripted timeline.", **plan)
    project.SetCurrentTimeline(timeline)

    appended_count = 0
    failed_clips: list[dict[str, Any]] = []
    append_payloads: list[dict[str, Any]] = []
    for clip in normalized_clips:
        item = item_by_path.get(clip["path"])
        if not item:
            failed_clips.append({"path": clip["path"], "reason": "media item unavailable after import"})
            continue
        payload: dict[str, Any] = {
            "mediaPoolItem": item,
            "startFrame": clip["start_frame"],
        }
        if clip["end_frame"] is not None:
            payload["endFrame"] = clip["end_frame"]
        media_type = _media_type_value(clip.get("media_type"))
        if media_type is not None:
            payload["mediaType"] = media_type
        if clip["track_index"] is not None:
            payload["trackIndex"] = clip["track_index"]
        if clip["record_frame"] is not None:
            payload["recordFrame"] = clip["record_frame"]
        append_payloads.append(payload)

    if append_payloads:
        appended = media_pool.AppendToTimeline(append_payloads) or []
        appended_count += len(appended)

    music_appended_count = 0
    for music_path in expanded_music_paths:
        item = item_by_path.get(music_path)
        if not item:
            failed_clips.append({"path": music_path, "reason": "music item unavailable after import"})
            continue
        payload = {"mediaPoolItem": item, "mediaType": 2, "trackIndex": 1}
        appended = media_pool.AppendToTimeline([payload]) or []
        music_appended_count += len(appended)

    added_markers = _add_markers_to_timeline(timeline, markers)

    return _ok(
        dry_run=False,
        timeline_name=timeline.GetName(),
        imported_count=len(imported_items),
        appended_count=appended_count,
        music_appended_count=music_appended_count,
        failed_clips=failed_clips,
        added_markers=added_markers,
        **plan,
    )


def render_timeline(
    target_dir: str,
    custom_name: str,
    preset_name: str | None = None,
    render_format: str | None = "mov",
    render_codec: str | None = "H264",
    render_settings: dict[str, Any] | None = None,
    start_render: bool = True,
    dry_run: bool = True,
    confirm: bool = False,
) -> dict[str, Any]:
    guard = _guard_mutation(dry_run, confirm, "Rendering the current timeline")
    if guard:
        return guard

    target = Path(target_dir).expanduser().resolve()
    settings = dict(render_settings or {})
    settings.update({"TargetDir": str(target), "CustomName": custom_name})
    plan = {
        "action": "render_timeline",
        "mode": "studio_live_control",
        "target_dir": str(target),
        "custom_name": custom_name,
        "preset_name": preset_name,
        "render_format": render_format,
        "render_codec": render_codec,
        "render_settings": settings,
        "start_render": start_render,
    }
    if dry_run:
        return _ok(dry_run=True, plan=plan)

    target.mkdir(parents=True, exist_ok=True)
    resolve, error = _connect()
    if error:
        return error
    project, error = _current_project(resolve)
    if error:
        return error
    if not project.GetCurrentTimeline():
        return _error("No current timeline is active.", **plan)

    preset_loaded = None
    if preset_name:
        preset_loaded = bool(project.LoadRenderPreset(preset_name))
    format_codec_set = None
    if render_format and render_codec:
        format_codec_set = bool(project.SetCurrentRenderFormatAndCodec(render_format, render_codec))
    render_settings_set = bool(project.SetRenderSettings(settings))
    job_id = project.AddRenderJob()
    if not job_id:
        return _error(
            "Resolve did not create a render job.",
            preset_loaded=preset_loaded,
            format_codec_set=format_codec_set,
            render_settings_set=render_settings_set,
            **plan,
        )

    started = False
    if start_render:
        started = bool(project.StartRendering(job_id))

    return _ok(
        dry_run=False,
        job_id=job_id,
        started=started,
        preset_loaded=preset_loaded,
        format_codec_set=format_codec_set,
        render_settings_set=render_settings_set,
        **plan,
    )


def render_status(job_id: str | None = None) -> dict[str, Any]:
    resolve, error = _connect()
    if error:
        return error
    project, error = _current_project(resolve)
    if error:
        return error

    status = project.GetRenderJobStatus(job_id) if job_id else None
    jobs = project.GetRenderJobList() or []
    return _ok(
        job_id=job_id,
        job_status=status,
        is_rendering=bool(project.IsRenderingInProgress()),
        render_jobs=jobs,
    )

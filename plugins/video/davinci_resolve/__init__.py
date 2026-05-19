"""Hermes plugin registration for DaVinci Resolve tools."""

from __future__ import annotations

import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

try:
    from . import schemas
    from .tools import (
        handle_append_to_current_timeline,
        handle_capabilities,
        handle_create_scripted_timeline,
        handle_create_timeline,
        handle_generate_fcpxml_timeline,
        handle_generate_marker_csv,
        handle_import_media,
        handle_launch,
        handle_project_summary,
        handle_probe,
        handle_render_status,
        handle_render_timeline,
        handle_scan_media_folder,
        handle_timeline_marker,
    )
except ImportError:
    import schemas
    from tools import (
        handle_append_to_current_timeline,
        handle_capabilities,
        handle_create_scripted_timeline,
        handle_create_timeline,
        handle_generate_fcpxml_timeline,
        handle_generate_marker_csv,
        handle_import_media,
        handle_launch,
        handle_project_summary,
        handle_probe,
        handle_render_status,
        handle_render_timeline,
        handle_scan_media_folder,
        handle_timeline_marker,
    )


TOOLS = [
    (
        "resolve_capabilities",
        schemas.RESOLVE_CAPABILITIES,
        handle_capabilities,
        "Explain available DaVinci Resolve tools, workflow order, and safety rules.",
    ),
    (
        "resolve_launch",
        schemas.RESOLVE_LAUNCH,
        handle_launch,
        "Open DaVinci Resolve on this computer and check scripting reachability.",
    ),
    (
        "resolve_probe",
        schemas.RESOLVE_PROBE,
        handle_probe,
        "Check whether DaVinci Resolve scripting is installed and reachable.",
    ),
    (
        "resolve_project_summary",
        schemas.RESOLVE_PROJECT_SUMMARY,
        handle_project_summary,
        "Inspect the current Resolve project and active timeline.",
    ),
    (
        "resolve_import_media",
        schemas.RESOLVE_IMPORT_MEDIA,
        handle_import_media,
        "Import media files into the Resolve media pool.",
    ),
    (
        "resolve_create_timeline",
        schemas.RESOLVE_CREATE_TIMELINE,
        handle_create_timeline,
        "Create a Resolve timeline, optionally seeded with imported media.",
    ),
    (
        "resolve_append_to_current_timeline",
        schemas.RESOLVE_APPEND_TO_CURRENT_TIMELINE,
        handle_append_to_current_timeline,
        "Append media files to the current Resolve timeline.",
    ),
    (
        "resolve_add_timeline_marker",
        schemas.RESOLVE_ADD_TIMELINE_MARKER,
        handle_timeline_marker,
        "Add a marker to the current Resolve timeline.",
    ),
    (
        "resolve_scan_media_folder",
        schemas.RESOLVE_SCAN_MEDIA_FOLDER,
        handle_scan_media_folder,
        "Scan a folder for video, audio, and image media before planning an edit.",
    ),
    (
        "resolve_create_scripted_timeline",
        schemas.RESOLVE_CREATE_SCRIPTED_TIMELINE,
        handle_create_scripted_timeline,
        "Create a Resolve Studio timeline from a structured scripted edit plan.",
    ),
    (
        "resolve_render_timeline",
        schemas.RESOLVE_RENDER_TIMELINE,
        handle_render_timeline,
        "Configure and start rendering the current Resolve Studio timeline.",
    ),
    (
        "resolve_render_status",
        schemas.RESOLVE_RENDER_STATUS,
        handle_render_status,
        "Check Resolve Studio render job progress.",
    ),
    (
        "resolve_generate_fcpxml_timeline",
        schemas.RESOLVE_GENERATE_FCPXML_TIMELINE,
        handle_generate_fcpxml_timeline,
        "Generate an FCPXML timeline that free DaVinci Resolve can import.",
    ),
    (
        "resolve_generate_marker_csv",
        schemas.RESOLVE_GENERATE_MARKER_CSV,
        handle_generate_marker_csv,
        "Generate a marker CSV for free Resolve interchange workflows.",
    ),
]


def register(ctx):
    for name, schema, handler, description in TOOLS:
        ctx.register_tool(
            name=name,
            toolset="davinciresolve",
            schema=schema,
            handler=handler,
            description=description,
        )

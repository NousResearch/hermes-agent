"""Tool schemas exposed to Hermes."""

from __future__ import annotations


DRY_RUN_FIELD = {
    "type": "boolean",
    "description": (
        "When true, report the planned Resolve action without modifying the project. "
        "Defaults to true for mutating tools."
    ),
    "default": True,
}

CONFIRM_FIELD = {
    "type": "boolean",
    "description": (
        "Must be true when dry_run is false. This prevents accidental project changes."
    ),
    "default": False,
}

MEDIA_PATHS_FIELD = {
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1,
    "description": "Absolute or user-relative file paths to video, audio, or image media.",
}


RESOLVE_PROBE = {
    "name": "resolve_probe",
    "description": (
        "Read-only diagnostic for DaVinci Resolve scripting. Use before other Resolve "
        "tools to confirm the module imports and a running Resolve app is reachable."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


RESOLVE_CAPABILITIES = {
    "name": "resolve_capabilities",
    "description": (
        "Return a concise operating guide for how an LLM agent should use this "
        "DaVinci Resolve plugin, including safe workflow order and available tools."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


RESOLVE_LAUNCH = {
    "name": "resolve_launch",
    "description": (
        "Open DaVinci Resolve or DaVinci Resolve Studio on the local Mac, wait briefly, "
        "and report whether the Resolve scripting API is reachable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "variant": {
                "type": "string",
                "enum": [
                    "auto",
                    "resolve",
                    "studio",
                    "beta",
                    "resolve20",
                    "studio20",
                    "resolve21",
                    "studio21",
                ],
                "description": (
                    "Which Resolve app to launch. Auto tries installed app paths first. "
                    "Use app_path when multiple versions are installed with custom names."
                ),
                "default": "auto",
            },
            "app_path": {
                "type": "string",
                "description": (
                    "Optional explicit path to a DaVinci Resolve .app bundle. Use this for "
                    "side-by-side Resolve 20 and Resolve 21 beta installs."
                ),
            },
            "wait_seconds": {
                "type": "integer",
                "minimum": 0,
                "maximum": 120,
                "description": "Seconds to wait before checking scripting reachability.",
                "default": 8,
            },
        },
        "additionalProperties": False,
    },
}


RESOLVE_PROJECT_SUMMARY = {
    "name": "resolve_project_summary",
    "description": (
        "Read-only summary of the active DaVinci Resolve project, current timeline, "
        "timeline count, frame rate, and marker count."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


RESOLVE_IMPORT_MEDIA = {
    "name": "resolve_import_media",
    "description": (
        "Import media files into the current DaVinci Resolve project's media pool. "
        "Use dry_run first, then call with dry_run=false and confirm=true after the user approves."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "paths": MEDIA_PATHS_FIELD,
            "bin_name": {
                "type": "string",
                "description": "Optional media pool bin name to create/use before importing.",
            },
            "dry_run": DRY_RUN_FIELD,
            "confirm": CONFIRM_FIELD,
        },
        "required": ["paths"],
        "additionalProperties": False,
    },
}


RESOLVE_CREATE_TIMELINE = {
    "name": "resolve_create_timeline",
    "description": (
        "Create a timeline in the current DaVinci Resolve project. Optionally import "
        "and seed the timeline with media files. Defaults to dry_run=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name for the new timeline.",
            },
            "media_paths": MEDIA_PATHS_FIELD,
            "dry_run": DRY_RUN_FIELD,
            "confirm": CONFIRM_FIELD,
        },
        "required": ["name"],
        "additionalProperties": False,
    },
}


RESOLVE_APPEND_TO_CURRENT_TIMELINE = {
    "name": "resolve_append_to_current_timeline",
    "description": (
        "Import media files if needed and append them to the active DaVinci Resolve timeline. "
        "Use for adding footage or music to the edit. Defaults to dry_run=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "media_paths": MEDIA_PATHS_FIELD,
            "dry_run": DRY_RUN_FIELD,
            "confirm": CONFIRM_FIELD,
        },
        "required": ["media_paths"],
        "additionalProperties": False,
    },
}


RESOLVE_ADD_TIMELINE_MARKER = {
    "name": "resolve_add_timeline_marker",
    "description": (
        "Add a marker to the current Resolve timeline at a frame number. Defaults to dry_run=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "frame": {
                "type": "integer",
                "minimum": 0,
                "description": "Timeline frame number for the marker.",
            },
            "color": {
                "type": "string",
                "description": "Resolve marker color, for example Blue, Green, Yellow, Red, or Purple.",
                "default": "Blue",
            },
            "name": {
                "type": "string",
                "description": "Marker name.",
            },
            "note": {
                "type": "string",
                "description": "Optional marker note.",
                "default": "",
            },
            "duration": {
                "type": "integer",
                "minimum": 1,
                "description": "Marker duration in frames.",
                "default": 1,
            },
            "dry_run": DRY_RUN_FIELD,
            "confirm": CONFIRM_FIELD,
        },
        "required": ["frame", "name"],
        "additionalProperties": False,
    },
}


MARKERS_FIELD = {
    "type": "array",
    "description": "Timeline markers to include in generated interchange artifacts.",
    "items": {
        "type": "object",
        "properties": {
            "frame": {
                "type": "integer",
                "minimum": 0,
                "description": "Timeline frame number.",
            },
            "name": {
                "type": "string",
                "description": "Marker name.",
            },
            "color": {
                "type": "string",
                "description": "Marker color, for example Blue, Green, Yellow, Red, or Purple.",
                "default": "Blue",
            },
            "note": {
                "type": "string",
                "description": "Optional marker note.",
                "default": "",
            },
            "duration": {
                "type": "integer",
                "minimum": 1,
                "description": "Duration in frames.",
                "default": 1,
            },
        },
        "required": ["frame", "name"],
        "additionalProperties": False,
    },
}


RESOLVE_GENERATE_FCPXML_TIMELINE = {
    "name": "resolve_generate_fcpxml_timeline",
    "description": (
        "Generate an FCPXML timeline file for free DaVinci Resolve interchange mode. "
        "Use this when live scripting is unavailable because the user has free Resolve."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Timeline/project name to write into the FCPXML.",
            },
            "media_paths": MEDIA_PATHS_FIELD,
            "output_path": {
                "type": "string",
                "description": "Optional destination .fcpxml path. Defaults to ~/Documents/Hermes Resolve Exports.",
            },
            "frame_rate": {
                "type": "integer",
                "minimum": 1,
                "maximum": 240,
                "description": "Timeline frame rate.",
                "default": 24,
            },
            "clip_duration_seconds": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Assumed duration for each clip when creating the interchange timeline.",
                "default": 5.0,
            },
            "width": {
                "type": "integer",
                "minimum": 1,
                "description": "Timeline width.",
                "default": 1920,
            },
            "height": {
                "type": "integer",
                "minimum": 1,
                "description": "Timeline height.",
                "default": 1080,
            },
            "markers": MARKERS_FIELD,
            "dry_run": DRY_RUN_FIELD,
        },
        "required": ["name", "media_paths"],
        "additionalProperties": False,
    },
}


CLIP_PLAN_FIELD = {
    "type": "array",
    "minItems": 1,
    "description": (
        "Structured edit decision list generated by Hermes from the user's script. "
        "Each item points to an imported media file and optional source/record ranges."
    ),
    "items": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or user-relative media path.",
            },
            "name": {
                "type": "string",
                "description": "Optional human-readable shot/beat name.",
            },
            "start_frame": {
                "type": "integer",
                "minimum": 0,
                "description": "Source start frame for this clip. Defaults to 0.",
                "default": 0,
            },
            "end_frame": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional source end frame. Omit to append from start_frame onward.",
            },
            "record_frame": {
                "type": "integer",
                "minimum": 0,
                "description": "Optional timeline frame where this clip should be placed.",
            },
            "media_type": {
                "type": "string",
                "enum": ["video", "audio"],
                "description": "Optional AppendToTimeline mediaType: video-only or audio-only.",
            },
            "track_index": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional target Resolve track index.",
            },
            "note": {
                "type": "string",
                "description": "Why Hermes selected this clip or which script beat it supports.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
}


RESOLVE_SCAN_MEDIA_FOLDER = {
    "name": "resolve_scan_media_folder",
    "description": (
        "Scan a local folder for video, audio, and still-image media so Hermes can plan "
        "a scripted edit from available source clips."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "folder_path": {
                "type": "string",
                "description": "Folder containing footage, audio, stills, or nested media folders.",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to scan subfolders.",
                "default": True,
            },
            "include_extensions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional extension allow-list, for example ['.mov', '.mp4', '.wav'].",
            },
            "include_durations": {
                "type": "boolean",
                "description": "When true, use ffprobe if available to include approximate media durations.",
                "default": False,
            },
            "max_files": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5000,
                "description": "Maximum number of media files to return.",
                "default": 500,
            },
        },
        "required": ["folder_path"],
        "additionalProperties": False,
    },
}


RESOLVE_CREATE_SCRIPTED_TIMELINE = {
    "name": "resolve_create_scripted_timeline",
    "description": (
        "Create a DaVinci Resolve Studio timeline from a structured edit plan. Hermes "
        "should derive the plan from the user's script, then dry-run this tool before "
        "actual project changes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Timeline name.",
            },
            "clips": CLIP_PLAN_FIELD,
            "music_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional music/audio files to import and append as audio.",
            },
            "markers": MARKERS_FIELD,
            "bin_name": {
                "type": "string",
                "description": "Optional media pool bin for imported edit assets.",
                "default": "Hermes Edit",
            },
            "dry_run": DRY_RUN_FIELD,
            "confirm": CONFIRM_FIELD,
        },
        "required": ["name", "clips"],
        "additionalProperties": False,
    },
}


RESOLVE_RENDER_TIMELINE = {
    "name": "resolve_render_timeline",
    "description": (
        "Configure a render job for the current Resolve Studio timeline and optionally "
        "start rendering a QuickTime/H.264-style output."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target_dir": {
                "type": "string",
                "description": "Output folder for the rendered file.",
            },
            "custom_name": {
                "type": "string",
                "description": "Render output base filename without extension.",
            },
            "preset_name": {
                "type": "string",
                "description": "Optional Resolve render preset to load before setting render options.",
            },
            "render_format": {
                "type": "string",
                "description": "Resolve render format key. Defaults to mov.",
                "default": "mov",
            },
            "render_codec": {
                "type": "string",
                "description": "Resolve render codec key. Defaults to H264.",
                "default": "H264",
            },
            "render_settings": {
                "type": "object",
                "description": (
                    "Optional Resolve SetRenderSettings dictionary. TargetDir and CustomName "
                    "are filled from target_dir and custom_name."
                ),
                "additionalProperties": True,
            },
            "start_render": {
                "type": "boolean",
                "description": "When true, start the render after creating the job.",
                "default": True,
            },
            "dry_run": DRY_RUN_FIELD,
            "confirm": CONFIRM_FIELD,
        },
        "required": ["target_dir", "custom_name"],
        "additionalProperties": False,
    },
}


RESOLVE_RENDER_STATUS = {
    "name": "resolve_render_status",
    "description": "Check Resolve Studio render progress and render queue state.",
    "parameters": {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "Optional render job id returned by resolve_render_timeline.",
            },
        },
        "additionalProperties": False,
    },
}


RESOLVE_GENERATE_MARKER_CSV = {
    "name": "resolve_generate_marker_csv",
    "description": (
        "Generate a marker CSV/manifest for free Resolve workflows when live marker "
        "insertion is unavailable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "markers": MARKERS_FIELD,
            "output_path": {
                "type": "string",
                "description": "Optional destination CSV path. Defaults to ~/Documents/Hermes Resolve Exports.",
            },
            "name": {
                "type": "string",
                "description": "Base name for the generated marker file.",
                "default": "Hermes Markers",
            },
            "frame_rate": {
                "type": "integer",
                "minimum": 1,
                "maximum": 240,
                "description": "Frame rate used to calculate seconds from frame numbers.",
                "default": 24,
            },
            "dry_run": DRY_RUN_FIELD,
        },
        "required": ["markers"],
        "additionalProperties": False,
    },
}

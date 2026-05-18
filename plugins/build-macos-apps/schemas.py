"""Schemas for the build-macos-apps plugin."""

MACOS_INSPECT_PROJECT_SCHEMA = {
    "name": "macos_inspect_project",
    "description": (
        "Inspect a local macOS app repository for Xcode build containers. "
        "Reports .xcworkspace/.xcodeproj files, Swift Package hints, and the "
        "recommended container to use for scheme listing or builds."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Repository or project path to inspect.",
            }
        },
        "required": ["path"],
    },
}

MACOS_LIST_SCHEMES_SCHEMA = {
    "name": "macos_list_schemes",
    "description": (
        "List Xcode schemes, targets, and configurations for a macOS project "
        "or workspace using xcodebuild -list -json."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Repository or project path that contains the Xcode container.",
            },
            "container_path": {
                "type": "string",
                "description": "Optional explicit .xcworkspace or .xcodeproj path to use.",
            },
        },
        "required": ["path"],
    },
}

MACOS_BUILD_PROJECT_SCHEMA = {
    "name": "macos_build_project",
    "description": (
        "Build a local macOS Xcode scheme with xcodebuild. This Phase 1 tool "
        "performs an unsigned build only; it does not run tests, launch apps, "
        "or handle notarization."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Repository or project path that contains the Xcode container.",
            },
            "scheme": {
                "type": "string",
                "description": "Xcode scheme to build.",
            },
            "container_path": {
                "type": "string",
                "description": "Optional explicit .xcworkspace or .xcodeproj path to use.",
            },
            "configuration": {
                "type": "string",
                "description": "Build configuration, usually Debug or Release.",
                "default": "Debug",
            },
            "destination": {
                "type": "string",
                "description": "xcodebuild destination string.",
                "default": "generic/platform=macOS",
            },
            "derived_data_path": {
                "type": "string",
                "description": "Optional DerivedData output path.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Maximum build time before Hermes aborts the command.",
                "default": 1800,
                "minimum": 30,
                "maximum": 7200,
            },
        },
        "required": ["path", "scheme"],
    },
}

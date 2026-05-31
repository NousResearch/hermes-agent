"""Tool schemas for the eikon plugin."""

from __future__ import annotations

EIKON_INSTALL_SCHEMA = {
    "name": "eikon_install",
    "description": (
        "Install a Herm eikon/avatar from the public catalog, a manifest URL, "
        "a git repository, or a local directory. Uses the existing `herm eikon install` "
        "path; no separate eikon executable is required."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Catalog name, HTTPS manifest/base URL, git URL, or local directory.",
            },
            "name": {
                "type": "string",
                "description": "Optional installed name override.",
            },
            "media": {
                "type": "boolean",
                "description": "Whether to fetch source media into the profile. Default true.",
                "default": True,
            },
            "no_source": {
                "type": "boolean",
                "description": "Alias for media=false.",
                "default": False,
            },
            "set_active": {
                "type": "boolean",
                "description": "Set the installed eikon as the active Herm avatar. Default true.",
                "default": True,
            },
        },
        "required": ["source"],
    },
}

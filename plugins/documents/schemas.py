"""Tool schema for the documents plugin.

Kept in its own module so that tests and the desktop IPC stub can import the
schema without dragging in the handler / network code.
"""

from __future__ import annotations

from typing import Any, Dict


REGISTER_DOCUMENT_SOURCE_SCHEMA: Dict[str, Any] = {
    "name": "register_document_source",
    "description": (
        "Register a file on disk as a companion document_source and "
        "return the newly-allocated UUID. Required before calling "
        "link_document_to_client, which takes a document_source UUID. "
        "The file MUST already exist on disk under a connected folder root "
        "(use scan_folder to discover candidates). Hermes never touches the "
        "desktop PGLite database directly — this tool calls the desktop's "
        "IPC endpoint `documents:register`, which runs the same "
        "upsertDocumentSource path that the folder-watch orchestrator uses."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": (
                    "Absolute path to the file on disk. Must exist and be "
                    "inside a connected folder-connector root (otherwise the "
                    "desktop will reject)."
                ),
            },
            "source_type": {
                "type": "string",
                "description": (
                    "Optional hint for the desktop ingester about the "
                    "document kind (e.g. 'contract', 'invoice', 'id_doc'). "
                    "Forwarded verbatim to the desktop IPC; the desktop is "
                    "the source of truth for valid values."
                ),
            },
            "metadata": {
                "type": "object",
                "description": (
                    "Optional free-form key/value metadata to attach to the "
                    "register call (e.g. client_hint, language). Forwarded "
                    "verbatim; the desktop decides which keys it persists."
                ),
                "additionalProperties": True,
            },
        },
        "required": ["file_path"],
        "additionalProperties": False,
    },
}

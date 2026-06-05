"""documents plugin — register filesystem files as companion document sources.

Exposes a single tool, ``register_document_source(file_path, source_type?,
metadata?)``, which the agent calls *before* ``link_document_to_client``.
The companion owns the PGLite ``document_sources`` table; this plugin is a
thin IPC client per the Hermes boundary policy.

Hebrew operator label: "רישום מסמך כמקור"  (category: documents).
"""

from __future__ import annotations

import logging

from plugins.documents.documents_plugin import (
    check_documents_requirements,
    handle_register_document_source,
)
from plugins.documents.schemas import REGISTER_DOCUMENT_SOURCE_SCHEMA


logger = logging.getLogger(__name__)


# Hebrew operator-facing label + category, mirroring the spec in the intake
# acceptance criteria. Surfaced via the manifest extras so the desktop /
# approval UI can display it.
LABEL_HE: str = "רישום מסמך כמקור"
CATEGORY: str = "documents"


_TOOLS = (
    (
        "register_document_source",
        REGISTER_DOCUMENT_SOURCE_SCHEMA,
        handle_register_document_source,
        "📄",
    ),
)


def register(ctx) -> None:
    """Register the register_document_source tool with the plugin context.

    Called once by the plugin loader when ``documents`` is listed in
    ``plugins.enabled`` (config.yaml).
    """
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="documents",
            schema=schema,
            handler=handler,
            check_fn=check_documents_requirements,
            emoji=emoji,
            description=(
                "Register a file as a companion document_source and return its "
                f"UUID. (label_he={LABEL_HE!r}, category={CATEGORY!r})"
            ),
        )
    logger.debug("documents plugin: registered %d tool(s)", len(_TOOLS))

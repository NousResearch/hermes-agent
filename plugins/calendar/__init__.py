"""calendar plugin — Google Calendar connector via gws subprocess wrapper.

Requires ``AGENTE_GWS_BIN`` env var pointing to the gws binary. OAuth and
token storage are owned entirely by gws; this plugin only shells out to the
CLI and parses its JSON output.

Provides 5 tools: list_events, create_event, update_event, cancel_event,
find_free_slots.
"""

from __future__ import annotations

import logging

from plugins.calendar import calendar_plugin as cp
from plugins.calendar.schemas import TOOL_SCHEMAS

logger = logging.getLogger(__name__)


def _check_gws() -> list[str] | None:
    """Preflight check — returns error strings or None if gws is available."""
    import os

    bin_path = os.environ.get("AGENTE_GWS_BIN")
    if not bin_path:
        return ["AGENTE_GWS_BIN environment variable is not set"]
    import shutil

    if shutil.which(bin_path) is not None or os.access(bin_path, os.X_OK):
        return None
    return [f"gws binary not found or not executable at {bin_path}"]


_TOOLS = (
    (
        "list_events",
        TOOL_SCHEMAS["list_events"],
        cp.list_events,
        "\U0001f4c5",
    ),
    (
        "create_event",
        TOOL_SCHEMAS["create_event"],
        cp.create_event,
        "\u2795",
    ),
    (
        "update_event",
        TOOL_SCHEMAS["update_event"],
        cp.update_event,
        "\u270f\ufe0f",
    ),
    (
        "cancel_event",
        TOOL_SCHEMAS["cancel_event"],
        cp.cancel_event,
        "\u274c",
    ),
    (
        "find_free_slots",
        TOOL_SCHEMAS["find_free_slots"],
        cp.find_free_slots,
        "\u23f0",
    ),
)


def register(ctx) -> None:
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="calendar",
            schema=schema,
            handler=handler,
            check_fn=_check_gws,
            emoji=emoji,
        )

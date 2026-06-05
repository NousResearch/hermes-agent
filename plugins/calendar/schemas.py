"""JSON-Schema definitions for the calendar plugin tools.

These mirror the ``gws calendar`` CLI surface. Each schema also carries a
``label_he`` (Hebrew label for operator-facing surfaces) and a ``category``
field consumed by the companion tool palette.
"""

from __future__ import annotations

from typing import Any, Dict


LIST_CALENDAR_EVENTS_SCHEMA: Dict[str, Any] = {
    "name": "list_calendar_events",
    "description": (
        "List Google Calendar events between two ISO-8601 timestamps. "
        "Shells `gws calendar list --from <after> --to <before> --json` and "
        "returns the parsed event list. OAuth is owned by gws; this tool "
        "never sees a token."
    ),
    "label_he": "רשימת אירועים ביומן",
    "category": "calendar",
    "parameters": {
        "type": "object",
        "properties": {
            "after": {
                "type": "string",
                "description": "ISO-8601 timestamp or `gws`-relative form (e.g. 'today', 'today+1d').",
            },
            "before": {
                "type": "string",
                "description": "ISO-8601 timestamp or `gws`-relative form (e.g. 'today+7d').",
            },
            "calendar_id": {
                "type": "string",
                "description": "Optional calendar id; defaults to primary.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 500,
                "description": "Optional max number of events to return.",
            },
        },
        "required": ["after", "before"],
        "additionalProperties": False,
    },
}


CREATE_CALENDAR_EVENT_SCHEMA: Dict[str, Any] = {
    "name": "create_calendar_event",
    "description": (
        "Create a new Google Calendar event. Shells "
        "`gws calendar create --title <title> --start <start> --end <end> "
        "[--attendees a,b] [--description <desc>] --json` and returns the "
        "created event id + html link. Hebrew/RTL titles are preserved verbatim."
    ),
    "label_he": "יצירת אירוע ביומן",
    "category": "calendar",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Event title (UTF-8; RTL preserved)."},
            "start": {"type": "string", "description": "ISO-8601 start timestamp."},
            "end": {"type": "string", "description": "ISO-8601 end timestamp."},
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of attendee email addresses.",
            },
            "description": {
                "type": "string",
                "description": "Optional event description / body.",
            },
        },
        "required": ["title", "start", "end"],
        "additionalProperties": False,
    },
}


GET_CALENDAR_EVENT_SCHEMA: Dict[str, Any] = {
    "name": "get_calendar_event",
    "description": (
        "Fetch a single Google Calendar event by id. Shells "
        "`gws calendar get --event <event_id> --json` and returns the parsed event."
    ),
    "label_he": "פרטי אירוע ביומן",
    "category": "calendar",
    "parameters": {
        "type": "object",
        "properties": {
            "event_id": {"type": "string", "description": "Google Calendar event id."},
        },
        "required": ["event_id"],
        "additionalProperties": False,
    },
}

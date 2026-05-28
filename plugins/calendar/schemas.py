"""OpenAI function-schema definitions for the 5 calendar tools."""

from __future__ import annotations

from typing import Any

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "list_events": {
        "name": "list_events",
        "description": (
            "List calendar events in a time range. Returns events from the "
            "operator's primary Google Calendar unless a specific calendar_id "
            "is provided."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start": {
                    "type": "string",
                    "description": "Start of the time range (ISO 8601 datetime or date).",
                },
                "end": {
                    "type": "string",
                    "description": "End of the time range (ISO 8601 datetime or date).",
                },
                "calendar_id": {
                    "type": "string",
                    "description": "Optional Google Calendar ID (defaults to primary).",
                },
            },
            "required": ["start", "end"],
        },
    },
    "create_event": {
        "name": "create_event",
        "description": (
            "Create a new calendar event. Returns the created event with its "
            "Google Calendar event ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start": {
                    "type": "string",
                    "description": "Event start time (ISO 8601 datetime).",
                },
                "end": {
                    "type": "string",
                    "description": "Event end time (ISO 8601 datetime).",
                },
                "title": {
                    "type": "string",
                    "description": "Event title (Hebrew/RTL supported via UTF-8).",
                },
                "description": {
                    "type": "string",
                    "description": "Optional event description.",
                },
                "location": {
                    "type": "string",
                    "description": "Optional event location.",
                },
                "calendar_id": {
                    "type": "string",
                    "description": "Optional Google Calendar ID (defaults to primary).",
                },
            },
            "required": ["start", "end", "title"],
        },
    },
    "update_event": {
        "name": "update_event",
        "description": (
            "Update an existing calendar event. Only the fields provided are "
            "changed — omitted fields are left unchanged."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "Google Calendar event ID to update.",
                },
                "start": {
                    "type": "string",
                    "description": "Updated start time (ISO 8601 datetime).",
                },
                "end": {
                    "type": "string",
                    "description": "Updated end time (ISO 8601 datetime).",
                },
                "title": {
                    "type": "string",
                    "description": "Updated event title.",
                },
                "description": {
                    "type": "string",
                    "description": "Updated event description.",
                },
                "location": {
                    "type": "string",
                    "description": "Updated event location.",
                },
                "calendar_id": {
                    "type": "string",
                    "description": "Optional Google Calendar ID (defaults to primary).",
                },
            },
            "required": ["event_id"],
        },
    },
    "cancel_event": {
        "name": "cancel_event",
        "description": (
            "Cancel (delete) a calendar event. Returns confirmation that the "
            "event was removed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "Google Calendar event ID to cancel.",
                },
                "calendar_id": {
                    "type": "string",
                    "description": "Optional Google Calendar ID (defaults to primary).",
                },
            },
            "required": ["event_id"],
        },
    },
    "find_free_slots": {
        "name": "find_free_slots",
        "description": (
            "Find free time slots within a window. Returns a list of available "
            "slots that are at least duration_minutes long."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "duration_minutes": {
                    "type": "integer",
                    "description": "Minimum slot duration in minutes (default 30).",
                    "default": 30,
                },
                "within": {
                    "type": "string",
                    "description": "Time window to search within (ISO 8601 datetime or date range).",
                },
                "calendar_id": {
                    "type": "string",
                    "description": "Optional Google Calendar ID (defaults to primary).",
                },
            },
            "required": ["within"],
        },
    },
}

"""Realtime function-tool schemas exposed to the speech-to-speech model.

Realtime tools use a flat shape: ``{type, name, description, parameters}`` (not
the chat-completions ``{type:"function", function:{...}}`` nesting). The handler
dispatches calls to these by ``name``.
"""

from __future__ import annotations

HERMES_AGENT_CONSULT = {
    "type": "function",
    "name": "hermes_agent_consult",
    "description": (
        "Delegate to the Hermes agent to answer a question or perform an action — "
        "lookups, calculations, files, web, or running tools. Use this for anything "
        "beyond small talk. Returns a short result to speak to the caller."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look into or do, phrased as a task.",
            }
        },
        "required": ["query"],
    },
}

LOOK_AT_SCREEN = {
    "type": "function",
    "name": "look_at_screen",
    "description": (
        "Look at what the caller is currently showing — their shared screen or "
        "camera — and answer a question about it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "What to determine from the image."},
            "source": {
                "type": "string",
                "enum": ["screen", "camera"],
                "description": "Which feed to look at; defaults to the shared screen.",
            },
            "scope": {
                "type": "string",
                "enum": ["live", "history"],
                "description": (
                    "'live' = the current frame (default); 'history' = recent "
                    "keyframes, to answer about something shown earlier."
                ),
            },
        },
        "required": ["question"],
    },
}

SHOW_TO_CALLER = {
    "type": "function",
    "name": "show_to_caller",
    "description": (
        "Generate an image from a text prompt and display it on the bot's own video "
        "tile so the caller can see it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "What image to create and show."},
            "count": {
                "type": "integer",
                "description": "How many images to show as a paced slideshow (1-3). Default 1.",
            },
        },
        "required": ["prompt"],
    },
}


CALL_ME_BACK = {
    "type": "function",
    "name": "call_me_back",
    "description": (
        "Place an outbound Teams call back to the current caller to deliver a "
        "result. Use when work will take a while and the caller asked to be called "
        "back, or when ending the call but a result is still pending. The result is "
        "spoken once they answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The result/message to speak when they answer.",
            }
        },
        "required": ["message"],
    },
}


HERMES_AGENT_TASK = {
    "type": "function",
    "name": "hermes_agent_task",
    "description": (
        "Run a long-running job in the background (multi-step work or research that "
        "takes more than a few seconds). Acknowledge to the caller that you're on it; "
        "the result is delivered by calling them back when it's done. Use this instead "
        "of hermes_agent_consult when the work won't finish within the conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The task to run in the background."}
        },
        "required": ["query"],
    },
}


POST_MEETING_MINUTES = {
    "type": "function",
    "name": "post_meeting_minutes",
    "description": (
        "Summarize the meeting so far and post the minutes (key points, decisions, "
        "action items) to the Teams chat. Use when the caller asks to 'summarize the "
        "meeting' or send notes."
    ),
    "parameters": {"type": "object", "properties": {}},
}


def default_tools() -> list[dict]:
    return [
        HERMES_AGENT_CONSULT,
        HERMES_AGENT_TASK,
        LOOK_AT_SCREEN,
        SHOW_TO_CALLER,
        CALL_ME_BACK,
        POST_MEETING_MINUTES,
    ]

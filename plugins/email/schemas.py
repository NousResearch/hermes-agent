"""Tool argument schemas for the bundled email plugin."""

COMMON_STRING = {"type": "string"}

LIST_EMAILS_SCHEMA = {
    "name": "list_emails",
    "description": "List recent Gmail messages in a folder (default INBOX). Returns gws JSON envelope.",
    "parameters": {
        "type": "object",
        "properties": {
            "folder": {"type": "string", "description": "Gmail folder / label (default INBOX)"},
            "limit": {"type": "integer", "description": "Max messages to return (default 10)"},
        },
    },
}

READ_EMAIL_SCHEMA = {
    "name": "read_email",
    "description": "Read a single Gmail message by id. Body is returned raw (no bidi normalization) so Hebrew/RTL round-trips faithfully.",
    "parameters": {
        "type": "object",
        "properties": {"message_id": COMMON_STRING},
        "required": ["message_id"],
    },
}

DRAFT_REPLY_SCHEMA = {
    "name": "draft_reply",
    "description": "Create a Gmail draft replying to message_id with body.",
    "parameters": {
        "type": "object",
        "properties": {
            "message_id": COMMON_STRING,
            "body": COMMON_STRING,
        },
        "required": ["message_id", "body"],
    },
}

SEND_EMAIL_SCHEMA = {
    "name": "send_email",
    "description": "Send a new email via Gmail.",
    "parameters": {
        "type": "object",
        "properties": {
            "to": COMMON_STRING,
            "subject": COMMON_STRING,
            "body": COMMON_STRING,
        },
        "required": ["to", "subject", "body"],
    },
}

MARK_EMAIL_SCHEMA = {
    "name": "mark_email",
    "description": "Add or remove a Gmail label on a message (e.g. UNREAD, STARRED, INBOX).",
    "parameters": {
        "type": "object",
        "properties": {
            "message_id": COMMON_STRING,
            "add_label": COMMON_STRING,
            "remove_label": COMMON_STRING,
        },
        "required": ["message_id"],
    },
}

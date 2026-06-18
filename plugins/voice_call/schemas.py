"""Tool schema for the voice_call plugin."""

VOICE_CALL_SCHEMA = {
    "name": "voice_call",
    "description": (
        "Place and manage guarded phone calls through the configured Hermes voice service. "
        "Actions: call(to, purpose, context, escalation_policy), hangup(call_id), "
        "transfer_to_jason(call_id), get_transcript(call_id). Use only when the user "
        "explicitly asks Hermes to make or manage a voice call. The call action requires "
        "a concrete purpose and recipient context so Hermes can identify itself and why it is calling."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["call", "hangup", "transfer_to_jason", "get_transcript"],
                "description": "Operation to perform.",
            },
            "to": {
                "type": "string",
                "description": "Destination phone number for action=call. E.164 is preferred.",
            },
            "purpose": {
                "type": "string",
                "description": "Short, explicit reason for the call. Required for action=call.",
            },
            "context": {
                "type": "string",
                "description": "Recipient-specific context Hermes may state on the call. Required for action=call.",
            },
            "escalation_policy": {
                "type": "string",
                "enum": ["no_escalation", "transfer_on_request", "transfer_if_blocked", "take_message"],
                "description": "How Hermes may escalate if the recipient asks for Jason or the call is blocked.",
            },
            "call_id": {
                "type": "string",
                "description": "Call/session id for non-call actions.",
            },
        },
        "required": ["action"],
    },
}

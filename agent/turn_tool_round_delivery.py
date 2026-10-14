"""Remove private content from protected tool-call transcript rows."""


def redact_protected_tool_row(agent, assistant_msg):
    if getattr(agent, "_pre_delivery_gate_active", False) is True:
        assistant_msg["content"] = ""
        for key in ("reasoning", "reasoning_content", "api_content"):
            assistant_msg.pop(key, None)

"""Tool schemas for the llm-switch plugin."""

SWITCH_LOCAL_LLM = {
    "name": "switch_local_llm",
    "description": (
        "Switch the local LLM server to a different model. "
        "Only one model can run at a time on a single GPU. "
        "The server is automatically managed — calling this tool kills the "
        "current server (if any) and starts a new one with the requested model. "
        "Available actions: provide a model name to switch, 'status' to check "
        "what's running, 'stop' to kill the server, or 'list' to see available models."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": (
                    "Model name to switch to (must match a key in models.yaml), "
                    "or 'status' to check current server, 'stop' to kill it, "
                    "'list' to show available models."
                ),
            },
        },
        "required": ["model"],
    },
}

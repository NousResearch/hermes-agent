import os


CURSOR_API_URL = "https://api2.cursor.sh"
CURSOR_CLIENT_VERSION = os.getenv("CURSOR_CLIENT_VERSION", "cli-2026.01.09-231024f")
CURSOR_AGENT_RUN_PATH = "/agent.v1.AgentService/Run"
CURSOR_GET_USABLE_MODELS_PATH = "/agent.v1.AgentService/GetUsableModels"
CONNECT_END_STREAM_FLAG = 0b00000010

# Hermes historically used a `cursor-` prefix that Cursor's API rejects.
CURSOR_MODEL_ID_ALIASES = {
    "cursor-composer-2.5": "composer-2.5",
    "cursor-composer": "composer-2.5",
}


def normalize_cursor_model_id(model_id: str) -> str:
    normalized = (model_id or "").strip()
    return CURSOR_MODEL_ID_ALIASES.get(normalized, normalized)

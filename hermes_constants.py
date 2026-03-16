import os
from pathlib import Path

"""Shared constants for Hermes Agent.

Import-safe module with no dependencies — can be imported from anywhere
without risk of circular imports.
"""

def get_hermes_home() -> Path:
    """
    Determines the agent's base directory, prioritizing the HERMES_HOME 
    environment variable with a fallback to the default ~/.hermes directory.
    """
    env_path = os.environ.get("HERMES_HOME")
    if env_path:
        return Path(env_path).resolve()
    return Path.home() / ".hermes"

# Central constant used for all persistent data and configuration paths
HERMES_HOME = get_hermes_home()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL}/chat/completions"

NOUS_API_BASE_URL = "https://inference-api.nousresearch.com/v1"
NOUS_API_CHAT_URL = f"{NOUS_API_BASE_URL}/chat/completions"

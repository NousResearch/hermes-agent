"""Back-compat shim for env loading helpers.

Prefer importing from `agent.env_loader`.
"""

from agent.env_loader import (  # noqa: F401
    DEFAULT_DOTENV_ENCODINGS,
    load_dotenv_with_fallback,
    read_env_text_with_fallback,
    read_text_with_fallback,
)

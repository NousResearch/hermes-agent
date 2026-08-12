"""Request-local profile scope shared by API server handlers."""

from contextvars import ContextVar
from typing import Optional


# Profile selected by the /p/<profile>/ URL prefix for the current request.
# Set by the API-server middleware and read by handlers / background-run setup.
api_request_profile: ContextVar[Optional[str]] = ContextVar(
    "api_server_request_profile", default=None
)

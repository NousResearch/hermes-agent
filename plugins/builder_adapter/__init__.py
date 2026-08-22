"""Hermes-owned governed builder dispatch adapter.

The package is inert until an operator explicitly constructs and starts the
Unix-domain-socket service. Importing it never opens a socket, starts a worker,
or reads credentials.
"""

from .adapter import BuilderDispatchAdapter
from .auth import HMACAuthenticator, PrincipalKey, darwin_peer_credentials
from .errors import AdapterError
from .models import DispatchRequest
from .plugin_tools import TOOLS

__all__ = [
    "AdapterError",
    "BuilderDispatchAdapter",
    "DispatchRequest",
    "HMACAuthenticator",
    "PrincipalKey",
    "darwin_peer_credentials",
]


def register(ctx) -> None:
    """Register only the governed builder toolset; never auto-start service."""
    for name, schema, handler in TOOLS:
        ctx.register_tool(
            name=name,
            toolset="builder_adapter",
            schema=schema,
            handler=handler,
            emoji="🔒",
        )

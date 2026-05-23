"""Per-provider runtime credential resolvers.

Each module in this package owns the *resolution* logic for one
provider — composing the disk/refresh primitives in
:mod:`hermes_cli.auth` into a single function that the rest of the
codebase calls. Provider-specific store ownership contracts live in
the per-provider docstrings.
"""

from agent.auth.codex import (
    CodexCredentialSource,
    CodexCredentials,
    resolve_codex_credentials,
)

__all__ = [
    "CodexCredentialSource",
    "CodexCredentials",
    "resolve_codex_credentials",
]

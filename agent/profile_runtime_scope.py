"""Neutral profile runtime scope shared by gateway and delegated children.

The scope is context-local: Hermes home, profile secrets and terminal policy are
bound for the current execution context and restored in reverse order. This
module deliberately does not import gateway lifecycle code, so tools can use the
same scope without a gateway/tools circular dependency.
"""
from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import AsyncIterator, Iterator, Mapping, Optional


def load_profile_secret_scope(profile_home: Path) -> dict[str, str]:
    """Hydrate and load one profile's secrets without mutating process env."""
    from agent.secret_scope import build_profile_secret_scope
    from hermes_cli.env_loader import hydrate_profile_secret_sources
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_token = set_hermes_home_override(str(profile_home))
    try:
        hydrate_profile_secret_sources(Path(profile_home))
        return build_profile_secret_scope(Path(profile_home))
    finally:
        reset_hermes_home_override(home_token)


@contextmanager
def profile_runtime_scope(
    profile_home: Path,
    prepared_secret_scope: Optional[Mapping[str, str]] = None,
    *,
    hydrate_secrets: bool = True,
) -> Iterator[None]:
    """Bind a profile's home, secrets and terminal policy for one execution.

    All state is context-local and all tokens are restored on success, error,
    cancellation and timeout. A caller may provide a prepared secret mapping when
    secret hydration already happened off the event loop.
    """
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.terminal_scope import install_and_reset_profile_terminal_scope

    home = Path(profile_home)
    home_token = set_hermes_home_override(str(home))
    secret_token = None
    try:
        if prepared_secret_scope is not None:
            secrets = dict(prepared_secret_scope)
        elif hydrate_secrets:
            secrets = load_profile_secret_scope(home)
        else:
            from agent.secret_scope import build_profile_secret_scope
            secrets = build_profile_secret_scope(home)
        secret_token = set_secret_scope(secrets)
        try:
            with install_and_reset_profile_terminal_scope(home):
                yield
        finally:
            if secret_token is not None:
                reset_secret_scope(secret_token)
    finally:
        reset_hermes_home_override(home_token)


@asynccontextmanager
async def async_profile_runtime_scope(profile_home: Path) -> AsyncIterator[None]:
    """Async form that hydrates external secret sources off the event loop."""
    import asyncio

    secrets = await asyncio.to_thread(load_profile_secret_scope, Path(profile_home))
    with profile_runtime_scope(Path(profile_home), secrets, hydrate_secrets=False):
        yield

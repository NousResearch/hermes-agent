"""Profile runtime scope for the gateway runner, built off the event loop.

Two asynchronous seams bind a profile's runtime scope: the handoff watcher (once per tick) and
every routed turn. Both used to resolve the profile's files ON the gateway's event loop — ``.env``
plus hydrated external sources for the credentials, and ``.env`` + ``config.yaml`` for the terminal
policy. A stalled read there holds every other task: a wedged read once tripped three 30 s liveness
probes and ``gateway.shutdown_watchdog`` dumped the stacks and exited the gateway with 75
(#144883).

The async seam therefore builds the entire scope in a worker thread and keeps only the
``ContextVar`` writes on the loop, where the body runs and where they are undone. The synchronous
``_profile_runtime_scope`` keeps working exactly as before and additionally accepts a prebuilt
terminal policy, so the async path can hand its work over instead of rebuilding it.

``_load_profile_secret_scope`` stays in ``gateway/run.py`` (tests monkeypatch it there), so it is
resolved through ``gateway.run`` at call time rather than imported at module load.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional


def _build_profile_runtime_scope(profile_home: Path) -> tuple[dict, Any]:
    """The file-backed half of :func:`_profile_runtime_scope`, for a worker thread.

    Both halves read the profile's files. Async callers run this through ``asyncio.to_thread`` and
    hand the result back as ``prepared_secret_scope`` / ``prepared_terminal_scope``, so a stalled
    disk read stalls a worker instead of the event loop.
    """
    from gateway.run import _load_profile_secret_scope
    from tools.terminal_scope import build_profile_terminal_scope_or_refusal

    home = Path(profile_home)
    return (
        _load_profile_secret_scope(home),
        build_profile_terminal_scope_or_refusal(home),
    )


@contextmanager
def _profile_runtime_scope(
    profile_home: Path, prepared_secret_scope: Optional[dict] = None, *,
    hydrate_secrets: bool = True, prepared_terminal_scope: Optional[tuple] = None):
    """Scope config/skills/memory AND credentials to a profile for one turn (multiplexed path only).
    ``set_hermes_home_override`` is a contextvar (reaches the agent worker via ``copy_context()``);
    ``set_secret_scope`` makes the profile ``.env`` the credential source without mutating
    ``os.environ``, so subprocesses never inherit cross-profile secrets.

    ``prepared_terminal_scope`` is the policy built by
    ``tools.terminal_scope.build_profile_terminal_scope_or_refusal`` — pass it when the caller
    already resolved the profile's files off-loop (``_async_profile_runtime_scope``)."""
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from gateway.run import _load_profile_secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_token = secret_token = None
    try:
        home_token = set_hermes_home_override(str(profile_home))
        if prepared_secret_scope is not None:
            secrets = prepared_secret_scope
        elif hydrate_secrets:
            secrets = _load_profile_secret_scope(Path(profile_home))
        else:
            from tui_gateway.launch_profile_policy import owner_secret_scope  # caller already hydrated off-loop
            secrets = owner_secret_scope(Path(profile_home))
        secret_token = set_secret_scope(secrets, profile_home=str(profile_home))
        # Install the routed profile's COMPLETE terminal policy, never ambient TERMINAL_* a prior turn set.
        # Without it terminal_tool reads the process-global TERMINAL_* vars a previous profile's turn may have
        # pinned (first-writer-wins backend leak; #68559).
        from tools.terminal_scope import (
            install_and_reset_prebuilt_terminal_scope,
            install_and_reset_profile_terminal_scope,
        )

        terminal_scope_cm = (
            install_and_reset_prebuilt_terminal_scope(prepared_terminal_scope)
            if prepared_terminal_scope is not None
            else install_and_reset_profile_terminal_scope(Path(profile_home))
        )
        with terminal_scope_cm:
            yield
    finally:
        if secret_token is not None:
            reset_secret_scope(secret_token)
        if home_token is not None:
            reset_hermes_home_override(home_token)


@asynccontextmanager
async def _async_profile_runtime_scope(profile_home: Path):
    """Enter a profile scope without loading its config/secret files on the event loop."""
    home = Path(profile_home)
    secrets, terminal_policy = await asyncio.to_thread(_build_profile_runtime_scope, home)
    with _profile_runtime_scope(home, secrets, prepared_terminal_scope=terminal_policy):
        yield

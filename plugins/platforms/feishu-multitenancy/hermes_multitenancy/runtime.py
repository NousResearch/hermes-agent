"""ProfileRuntime runs an agent inside a profile-specific HERMES_HOME context.

The active profile is carried in a ``ContextVar`` so concurrent asyncio tasks
cannot observe each other's profile home. ``HERMES_HOME`` is also switched
under a per-event-loop lock for compatibility with Hermes modules that still
read the environment directly.

The actual AIAgent invocation stays abstract (via ``run_agent_fn``) so tests
can inject mocks without spinning up real LLMs.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

HERMES_HOME_ENV = "HERMES_HOME"

# Authoritative per-task profile context.
_PROFILE_HOME_VAR: contextvars.ContextVar[Optional[Path]] = contextvars.ContextVar(
    "hermes_multitenancy_profile_home", default=None
)


def get_current_profile_home() -> Optional[Path]:
    """Return the active profile_home for the current asyncio task, or None."""
    return _PROFILE_HOME_VAR.get()


# Serializes os.environ HERMES_HOME mutations across concurrent dispatches.
# We key locks by event loop because asyncio.Lock binds to the loop on first
# acquire — pytest spawns a fresh loop per test so a module-level lock would
# raise "bound to a different event loop" on the second test. Production
# runs a single loop so this dict stays at 1 entry.
_ENV_LOCKS: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}


def _get_env_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _ENV_LOCKS.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _ENV_LOCKS[loop] = lock
    return lock


# Type for an agent-runner: takes the event + profile_home, returns a response string
AgentRunner = Callable[[Any, Path], Awaitable[str]]


# In-memory routing table for tests and non-persistent local demos.
_IN_MEMORY_ROUTING: dict[str, Path] = {}


def add_in_memory_route(open_id: str, profile_home: Path) -> None:
    """Register a hard-coded route for tests or local demos."""
    _IN_MEMORY_ROUTING[open_id] = profile_home


def clear_in_memory_routes() -> None:
    """Clear all in-memory routes."""
    _IN_MEMORY_ROUTING.clear()


def resolve_profile_home(open_id: str) -> Optional[Path]:
    """Resolve open_id to a profile home directory using in-memory routes."""
    return _IN_MEMORY_ROUTING.get(open_id)


class ProfileRuntime:
    """Runs an agent against a specific profile home directory.

    Use as ``await ProfileRuntime(profile_home).dispatch(event)``.

    Concurrency model:
      * The ContextVar is set BEFORE acquiring the env lock so the runner sees
        its profile_home immediately, even if env mutation is queued behind
        another dispatch.
      * The env lock is held for the entire body of the runner so any code
        inside the runner that reads ``os.environ['HERMES_HOME']`` (e.g. a
        legacy hermes module) sees the right value.

    Trade-off: code that depends on ``HERMES_HOME`` is serialized while the
    profile environment is active. The AIAgent subprocess path avoids sharing
    this state across concurrent tenant runs.
    """

    def __init__(
        self,
        profile_home: Path,
        run_agent_fn: Optional[AgentRunner] = None,
    ) -> None:
        self.profile_home = Path(profile_home)
        self._run_agent_fn = run_agent_fn or _default_run_agent

    async def dispatch(self, event: Any) -> str:
        """Set per-task profile context, run the agent under env lock."""
        token = _PROFILE_HOME_VAR.set(self.profile_home)
        try:
            async with _get_env_lock():
                original = os.environ.get(HERMES_HOME_ENV)
                os.environ[HERMES_HOME_ENV] = str(self.profile_home)
                try:
                    self._verify_switch()
                    return await self._run_agent_fn(event, self.profile_home)
                finally:
                    if original is None:
                        os.environ.pop(HERMES_HOME_ENV, None)
                    else:
                        os.environ[HERMES_HOME_ENV] = original
        finally:
            _PROFILE_HOME_VAR.reset(token)

    def _verify_switch(self) -> None:
        """Sanity-check that the switch reached both ContextVar and hermes_constants."""
        ctx = get_current_profile_home()
        if ctx != self.profile_home:
            logger.warning(
                "multitenancy: ContextVar mismatch (want=%s got=%s)",
                self.profile_home,
                ctx,
            )

        try:
            from hermes_constants import get_hermes_home  # type: ignore
        except ImportError:
            logger.debug("hermes_constants not importable (ok in unit tests)")
            return
        actual = get_hermes_home()
        if actual != self.profile_home:
            logger.warning(
                "multitenancy: HERMES_HOME env mismatch (want=%s got=%s)",
                self.profile_home,
                actual,
            )


async def _default_run_agent(event: Any, profile_home: Path) -> str:
    """Fallback echo runner used when no concrete runner is injected."""
    text = getattr(event, "text", "") or ""
    return f"[profile={profile_home.name}] echo: {text}"

"""
Per-turn agent context — lets a single gateway process serve multiple
profiles (memory/skills/soul) without rebinding the process-wide
``HERMES_HOME`` env var.

The mechanism is a single ``ContextVar[Optional[Path]]``.  The gateway
runtime sets this contextvar before invoking each ``AIAgent``, and
``hermes_constants.get_hermes_home()`` consults it first so that every
profile-aware path resolution (memory, skills, soul, sessions if scoped,
…) automatically picks up the active agent's home directory.

Design constraints:

- **Import-safe.** No transitive imports of anything that calls
  ``get_hermes_home()`` at module load time, otherwise the override path
  in ``hermes_constants`` would create a circular import.
- **Thread/Executor safe.** ``contextvars`` propagate through
  ``asyncio.to_thread`` / ``run_in_executor`` automatically when the
  caller uses ``contextvars.copy_context()`` (which the gateway already
  does in its background work paths).  Bare ``threading.Thread`` does NOT
  propagate; callers spawning raw threads must capture and re-set via
  ``current_agent_home()`` themselves.
- **Backward compatible.** When the contextvar is unset (the default),
  ``get_hermes_home()`` falls back to the existing ``HERMES_HOME`` env
  var path — so single-profile gateways and CLI invocations behave
  exactly as before.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


_AGENT_HOME: ContextVar[Optional[Path]] = ContextVar(
    "hermes_agent_home", default=None
)


def current_agent_home() -> Optional[Path]:
    """Return the active agent's HERMES_HOME, or ``None`` if unset.

    Consulted from ``hermes_constants.get_hermes_home()`` to override
    the process-wide env var when a gateway turn is running on behalf
    of a specific profile.
    """
    return _AGENT_HOME.get()


@contextmanager
def agent_home_scope(home: Path) -> Iterator[Path]:
    """Run a block with ``current_agent_home()`` set to ``home``.

    Restores the previous value on exit (supports nesting, e.g. an
    orchestrator agent that delegates to a sub-agent on a different
    profile).
    """
    resolved = Path(home)
    token = _AGENT_HOME.set(resolved)
    try:
        yield resolved
    finally:
        _AGENT_HOME.reset(token)


def reset_agent_home() -> None:
    """Clear the active agent home (back to env-var fallback).

    Intended for tests and gateway shutdown.  Production callers should
    use ``agent_home_scope`` so the previous value is restored
    automatically.
    """
    _AGENT_HOME.set(None)

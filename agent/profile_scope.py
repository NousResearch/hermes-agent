"\"\"\"
Per-profile runtime scope — shared between gateway and cron.

The multiplexing gateway serves many profiles from one process. Each profile
has its own ``HERMES_HOME`` (config / skills / memory / SOUL / sessions) and
its own ``.env`` (provider keys and platform tokens), so a per-turn handler
must redirect BOTH before any work that touches either.

This module hosts the single context-manager both the gateway inbound path
and the cron scheduler wrap their work in. Hosting it here (not in
``gateway.run``) breaks the natural ``gateway → cron`` import direction so
the cron scheduler can enter the same scope without a circular import —
every consumer of this scope is symmetric, so the seam lives at the lowest
practical layer.

Design rationale: ``docs/design/multiplexing-gateway.md`` (Workstream A).
\"\"\"
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only import — the real import happens lazily inside the manager
    # to keep module-load order independent of hermes_constants / agent
    # import timing (both modules are loaded as part of normal startup).
    pass


@contextmanager
def profile_runtime_scope(profile_home: \"Path\"):
    \"\"\"Scope config/skills/memory AND credentials to a profile for one turn.

    Combines the two seams the multiplexer needs:

      1. ``set_hermes_home_override`` — redirects ``get_hermes_home()``
         (config, skills, memory, SOUL, sessions) to the profile's home.
         Contextvar, so it propagates into the agent worker thread via
         ``copy_context()``.

      2. ``set_secret_scope`` — installs the profile's ``.env`` secrets as
         the authoritative credential source, so ``get_secret`` reads this
         profile's keys and never the process-global ``os.environ`` (which
         in a multiplexer may hold another profile's values).

    Used by the multiplexed inbound path AND the cron scheduler. Single-
    profile gateways and single-profile cron deployments skip the scope
    entirely, so their behavior is unchanged. Loading the profile's
    ``.env`` here does NOT mutate ``os.environ`` —
    ``build_profile_secret_scope`` returns an isolated dict — which is
    what keeps subprocesses (MCP, kanban, no_agent cron scripts) from
    inheriting cross-profile secrets.

    Args:
        profile_home: Absolute path to the profile's HERMES_HOME
            (e.g. ``~/.hermes/profiles/yangyang/`` or ``~/.hermes/`` for
            the default profile).
    \"\"\"
    # Lazy imports: avoid pinning cron → hermes_constants / agent module
    # load order. gateway.run and cron.scheduler both call this manager
    # from their hot paths; whichever module is loaded first should not
    # force the other to be importable.
    from hermes_constants import (
        set_hermes_home_override,
        reset_hermes_home_override,
    )
    from agent.secret_scope import (
        build_profile_secret_scope,
        set_secret_scope,
        reset_secret_scope,
    )

    profile_home = Path(profile_home)
    home_token = set_hermes_home_override(str(profile_home))
    secret_token = set_secret_scope(build_profile_secret_scope(profile_home))
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


# ── introspection helper for tests / debugging ────────────────────────
def active_profile_home() -> \"Path | None\":
    \"\"\"Return the HERMES_HOME override currently active in this context, or None.
    \"\"\"
    from hermes_constants import get_hermes_home_override
    override = get_hermes_home_override()
    return Path(override) if override else None

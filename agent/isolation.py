"""Shared agent isolation contract.

The CLI, TUI, messaging gateway, and API server all construct ``AIAgent``
instances for user sessions.  Isolation — skipping auto-injected rules
(``AGENTS.md`` / ``SOUL.md`` / ``.cursorrules``) and memory — must behave
identically across every entry point, otherwise ``--ignore-rules`` and
``--safe-mode`` only affect a subset of surfaces.

Resolution precedence:

1. Explicitly passed flags (``ignore_rules`` / ``safe_mode``) when either is
   provided (not ``None``) — an explicit ``False`` deliberately overrides a
   process-level env var so callers can force isolation off;
2. environment variables ``HERMES_IGNORE_RULES`` / ``HERMES_SAFE_MODE``
   (``HERMES_SAFE_MODE`` implies ignore-rules);
3. default ``False``.

The CLI normalizes its flags into those env vars at startup
(``hermes_cli/main.py``), so entry points that construct agents inside an
already-started process can rely on the env path; direct callers can pass
explicit flags instead.
"""

from __future__ import annotations

from utils import env_var_enabled


def resolve_agent_isolation(
    *,
    ignore_rules: bool | None = None,
    safe_mode: bool | None = None,
) -> bool:
    """Return whether the shared agent-isolation contract is enabled.

    Callers apply this one decision to both ``skip_context_files`` and
    ``skip_memory`` because the isolation contract skips rules and memory
    together.

    Args:
        ignore_rules: explicit ``--ignore-rules``-style flag; ``None`` means
            "not provided" and falls through to the env/default chain.
        safe_mode: explicit ``--safe-mode``-style flag; implies
            ``ignore_rules`` when set.
    """
    if ignore_rules is not None or safe_mode is not None:
        isolated = bool(ignore_rules or safe_mode)
    else:
        isolated = env_var_enabled("HERMES_IGNORE_RULES") or env_var_enabled(
            "HERMES_SAFE_MODE"
        )
    return isolated

"""Profile-scoped credential reads for fork-added tools (multiplex fix 4).

Fork tools (Jira, Notion, Webflow, Slack posting, Drive) used to read their
credentials with ``os.getenv`` and "self-heal" a missing value by loading
``~/.hermes/.env`` into ``os.environ``. Under ``gateway.multiplex_profiles``
``os.environ`` is the **default** profile's environment, so both moves hand a
secondary profile the default bot's credentials.

These helpers keep the single-profile path exactly as it was and, when
multiplexing is active, resolve through ``agent.secret_scope.get_secret``:

- a scoped miss is a miss (never ``os.environ``);
- an unscoped read (``UnscopedSecretError``) is logged at WARNING with the
  tool and key, then treated as "not configured";
- the ``.env`` self-heal does not run, because it writes the default's env.

See docs/superpowers/specs/2026-11-16-multiplex-fork-hardening-design.md.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

from agent.secret_scope import (
    UnscopedSecretError,
    current_secret_scope,
    get_secret,
    is_multiplex_active,
)

logger = logging.getLogger(__name__)


def read_secret(name: str, *, tool: str, default: str = "") -> str:
    """Return credential *name* for the current profile, stripped.

    Multiplex off: ``os.getenv(name, default).strip()`` — today's behaviour.
    Multiplex on: the profile's secret scope only; an unscoped read logs a
    WARNING naming *tool* and *name* and returns *default*.
    """
    if not is_multiplex_active():
        return (os.getenv(name, default) or "").strip()
    try:
        val = get_secret(name, default)
    except UnscopedSecretError:
        logger.warning(
            "[%s] %s read with no profile scope under multiplex; "
            "treating as not configured",
            tool,
            name,
        )
        return default
    return (val or "").strip() if val is not None else default


def heal_dotenv(**kwargs) -> bool:
    """Load the Hermes ``.env`` into ``os.environ`` — single-profile only.

    Headless cron/delegation workers may never have loaded the dotenv, which
    is why the fork tools self-heal. Under multiplex ``os.environ`` belongs to
    the default profile, so loading into it is not a credential source for the
    profile that owns the call (and would leak into the default's env).
    Returns True when the loader ran.
    """
    if is_multiplex_active():
        return False
    try:
        from hermes_cli.env_loader import load_hermes_dotenv

        load_hermes_dotenv(**kwargs)
    except Exception:
        return False
    return True


def _process_profile_secret_keys() -> Optional[set]:
    """Non-global keys the process (default) profile's secret sources define.

    ``None`` when they cannot be enumerated — the caller then fails closed.
    """
    try:
        from agent.secret_scope import _is_global_env, build_profile_secret_scope
        from hermes_constants import get_process_hermes_home

        keys = build_profile_secret_scope(get_process_hermes_home()).keys()
        return {k for k in keys if not _is_global_env(k)}
    except Exception:
        logger.warning(
            "could not enumerate the default profile's secrets; stripping every "
            "non-global variable from the child environment",
            exc_info=True,
        )
        return None


def scoped_subprocess_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Child-process env for the profile that owns the current context.

    Multiplex off: ``dict(base or os.environ)`` — unchanged.
    Multiplex on: the process environment minus every secret the process
    (default) profile's ``.env`` / secret sources define, overlaid with the
    current profile's secret scope. Unscoped → no overlay. If the default's
    secrets cannot be enumerated, every non-global variable is stripped
    (fail closed) before the overlay.
    """
    env = dict(base) if base is not None else dict(os.environ)
    if not is_multiplex_active():
        return env
    keys = _process_profile_secret_keys()
    if keys is None:
        from agent.secret_scope import _is_global_env

        keys = {k for k in env if not _is_global_env(k)}
    for key in keys:
        env.pop(key, None)
    scope = current_secret_scope()
    if scope:
        env.update({k: str(v) for k, v in scope.items() if v is not None})
    return env

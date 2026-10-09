"""Profile-scoped registration of user-configured hooks.

Every long-lived agent runtime must register the user's shell hooks and
outbound webhooks at startup, or events configured in config.yaml silently
never fire for sessions driven through that backend. Registrations are keyed
by ``hermes_home_key()`` at call time because one gateway process can serve
multiple profiles. A home is marked complete only after registration finishes,
so concurrent callers for that profile wait rather than observing an early
no-op. Plugin discovery runs before config hooks because plugin block decisions
must win ties, matching gateway startup.

Consent semantics are owned by ``agent.shell_hooks`` (flag / env / config
opt-in, fail-closed on non-TTY stdin) and neither helper ever prompts on a
backend's piped stdio. Both registrations are idempotent and fail-soft: a
broken hook config must never take down a backend.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_ensured_lock = threading.Lock()
_completed: set[str] = set()
_inflight_locks: dict[str, threading.Lock] = {}


def reset_for_tests() -> None:
    """Clear profile-scoped registration state (test isolation only)."""
    with _ensured_lock:
        _completed.clear()
        _inflight_locks.clear()


def ensure_hooks_registered(
    cfg=None, *, accept_hooks: bool = False, home: str | Path | None = None
) -> None:
    """Register shell hooks + outbound webhooks once for the current profile home.

    *cfg* defaults to a fresh ``hermes_cli.config.load_config()`` read.
    *accept_hooks* is passed through to shell-hook registration — callers
    that own a CLI consent flag pass it; backend entry points keep the
    default ``False`` and let the helper resolve opt-in from env/config.
    *home* is an optional explicit profile home for callers that need to
    register outside the currently bound profile scope.

    Never raises. Repeat calls for a completed home are no-ops; callers for
    a home whose registration is in progress wait for that work to finish.
    """
    from hermes_constants import hermes_home_key

    home_key = hermes_home_key(home)
    with _ensured_lock:
        if home_key in _completed:
            return
        registration_lock = _inflight_locks.setdefault(home_key, threading.Lock())

    with registration_lock:
        with _ensured_lock:
            if home_key in _completed:
                return
        try:
            # Plugin block decisions win ties with declarative hook config.
            from hermes_cli.plugins import discover_plugins

            discover_plugins()
        except Exception:
            logger.debug("plugin discovery failed before hook registration", exc_info=True)
        try:
            if cfg is None:
                from hermes_cli.config import load_config

                cfg = load_config()
            from agent import outbound_webhooks, shell_hooks

            shell_hooks.register_from_config(cfg, accept_hooks=accept_hooks)
            outbound_webhooks.register_from_config(cfg)
        except Exception:
            logger.debug(
                "shell-hook / outbound-webhook registration failed at startup",
                exc_info=True,
            )
            return
        with _ensured_lock:
            _completed.add(home_key)

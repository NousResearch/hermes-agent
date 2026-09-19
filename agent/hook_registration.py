"""Once-per-process registration of user-configured hooks.

Every long-lived agent runtime must register the user's shell hooks and
outbound webhooks at startup, or events configured in config.yaml silently
never fire for sessions driven through that backend. The call sites:

* ``hermes --cli`` / oneshot — ``hermes_cli.main._prepare_agent_startup``
  (registers inline, predates this module)
* messaging gateway — ``gateway/run.py`` (registers inline)
* TUI stdio backend — ``tui_gateway.entry.main`` → ``tui_gateway.server``
* TUI WebSocket sidecar (dashboard chat / desktop) —
  ``tui_gateway.ws.handle_ws`` → ``tui_gateway.server``
* ``hermes serve`` / dashboard backend — ``hermes_cli.web_server._lifespan``

The two inline call sites predate this module and behave identically; they
can migrate to :func:`ensure_hooks_registered` later without behavior
change. Consent semantics are owned by ``agent.shell_hooks`` (flag / env /
config opt-in, fail-closed on non-TTY stdin) and neither helper ever
prompts on a backend's piped stdio. Both registrations are idempotent and
fail-soft: a broken hook config must never take down a backend.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_ensured_lock = threading.Lock()
_ensured = False


def reset_for_tests() -> None:
    """Clear the once-per-process guard (test isolation only)."""
    global _ensured
    with _ensured_lock:
        _ensured = False


def ensure_hooks_registered(cfg=None, *, accept_hooks: bool = False) -> None:
    """Register shell hooks + outbound webhooks exactly once per process.

    *cfg* defaults to a fresh ``hermes_cli.config.load_config()`` read.
    *accept_hooks* is passed through to shell-hook registration — callers
    that own a CLI consent flag pass it; backend entry points keep the
    default ``False`` and let the helper resolve opt-in from env/config.

    Never raises. Repeat calls are no-ops (the underlying registrations
    are independently idempotent too, so a caller that must bypass the
    guard can invoke ``agent.shell_hooks`` / ``agent.outbound_webhooks``
    directly, as the CLI and gateway already do).
    """
    global _ensured
    with _ensured_lock:
        if _ensured:
            return
        _ensured = True
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
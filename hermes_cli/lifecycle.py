"""Hermes lifecycle dispatch for first-party observers and plugins."""

from __future__ import annotations

import logging
from typing import Any, List

from agent.runtime_policy import is_authoritative

logger = logging.getLogger(__name__)


def _observe(hook_name: str, **kwargs: Any) -> None:
    try:
        from hermes_cli.observability import observe_lifecycle

        observe_lifecycle(hook_name, **kwargs)
    except Exception:
        logger.warning("Built-in observability hook failed", exc_info=True)


def _plugin_hooks(hook_name: str, **kwargs: Any) -> List[Any]:
    from hermes_cli import plugins

    return plugins.invoke_hook(hook_name, **kwargs)


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Notify first-party observers, then invoke compatibility plugin hooks."""
    _observe(hook_name, **kwargs)
    return _plugin_hooks(hook_name, **kwargs)


def has_hook(hook_name: str) -> bool:
    """Return whether a first-party observer or plugin consumes a hook."""
    try:
        from hermes_cli.observability import handles_hook

        if handles_hook(hook_name):
            return True
    except Exception:
        logger.warning("Unable to inspect built-in observability hooks", exc_info=True)

    from hermes_cli import plugins

    return plugins.has_hook(hook_name)


def finalize_session(**kwargs: Any) -> List[Any]:
    """Settle required authority before teardown; a failure never skips observers or cleanup."""
    from hermes_cli import plugins_authority as authority
    session_id = str(kwargs.get("session_id") or "")
    run_id = str(kwargs.pop("runtime_run_id", "") or "")
    required_policy = kwargs.pop("runtime_policy", None)
    receipt = None
    settlement_error = None
    resolved = authority.resolve_authoritative_run(run_id=run_id, session_id=session_id) if (run_id or session_id) else None
    if is_authoritative(required_policy) and (
        not resolved or authority.authoritative_run_policy(resolved) != required_policy
    ):
        settlement_error = RuntimeError("required authoritative run lease is absent or mismatched")
    if resolved:
        try:
            receipt = authority.finalize_authoritative_run(resolved, **kwargs)
        except BaseException as exc:
            settlement_error = exc
    try:
        _observe("on_session_finalize", **kwargs)
        if session_id:
            try:
                from agent import relay_runtime
                relay_runtime.SESSION_COORDINATOR.finalize_conversation(
                    profile_key=relay_runtime.current_profile_key(), session_id=session_id)
            except Exception:
                logger.warning("Core Relay session finalization failed", exc_info=True)
        results = _plugin_hooks("on_session_finalize", **kwargs)
    finally:
        if settlement_error is not None:
            raise settlement_error
    return ([receipt] if receipt is not None else []) + results

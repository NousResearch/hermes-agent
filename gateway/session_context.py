"""
Session-scoped context variables for the Hermes gateway.

Replaces the previous ``os.environ``-based session state
(``HERMES_SESSION_PLATFORM``, ``HERMES_SESSION_CHAT_ID``, etc.) with
Python's ``contextvars.ContextVar``.

**Why this matters**

The gateway processes messages concurrently via ``asyncio``.  When two
messages arrive at the same time the old code did:

    os.environ["HERMES_SESSION_THREAD_ID"] = str(context.source.thread_id)

Because ``os.environ`` is *process-global*, Message A's value was
silently overwritten by Message B before Message A's agent finished
running.  Background-task notifications and tool calls therefore routed
to the wrong thread.

``contextvars.ContextVar`` values are *task-local*: each ``asyncio``
task (and any ``run_in_executor`` thread it spawns) gets its own copy,
so concurrent messages never interfere.

**Backward compatibility**

The public helper ``get_session_env(name, default="")`` mirrors the old
``os.getenv("HERMES_SESSION_*", ...)`` calls.  Existing tool code only
needs to replace the import + call site:

    # before
    import os
    platform = os.getenv("HERMES_SESSION_PLATFORM", "")

    # after
    from gateway.session_context import get_session_env
    platform = get_session_env("HERMES_SESSION_PLATFORM", "")
"""

import os
from contextvars import ContextVar
from typing import Any, Optional

# Sentinel to distinguish "never set in this context" from "explicitly set to empty".
# When a contextvar holds _UNSET, we fall back to os.environ (CLI/cron compat).
# When it holds "" (after clear_session_vars resets it), we return "" — no fallback.
_UNSET: Any = object()

# ---------------------------------------------------------------------------
# Per-task session variables
# ---------------------------------------------------------------------------

_SESSION_PLATFORM: ContextVar = ContextVar("HERMES_SESSION_PLATFORM", default=_UNSET)
_SESSION_CHAT_ID: ContextVar = ContextVar("HERMES_SESSION_CHAT_ID", default=_UNSET)
_SESSION_CHAT_NAME: ContextVar = ContextVar("HERMES_SESSION_CHAT_NAME", default=_UNSET)
_SESSION_THREAD_ID: ContextVar = ContextVar("HERMES_SESSION_THREAD_ID", default=_UNSET)
_SESSION_USER_ID: ContextVar = ContextVar("HERMES_SESSION_USER_ID", default=_UNSET)
_SESSION_USER_NAME: ContextVar = ContextVar("HERMES_SESSION_USER_NAME", default=_UNSET)
_SESSION_KEY: ContextVar = ContextVar("HERMES_SESSION_KEY", default=_UNSET)
_SESSION_ID: ContextVar = ContextVar("HERMES_SESSION_ID", default=_UNSET)

# Cron auto-delivery vars — set per-job in run_job() so concurrent jobs
# don't clobber each other's delivery targets.
_CRON_AUTO_DELIVER_PLATFORM: ContextVar = ContextVar("HERMES_CRON_AUTO_DELIVER_PLATFORM", default=_UNSET)
_CRON_AUTO_DELIVER_CHAT_ID: ContextVar = ContextVar("HERMES_CRON_AUTO_DELIVER_CHAT_ID", default=_UNSET)
_CRON_AUTO_DELIVER_THREAD_ID: ContextVar = ContextVar("HERMES_CRON_AUTO_DELIVER_THREAD_ID", default=_UNSET)

# Per-task terminal working directory.  The legacy env var is ``TERMINAL_CWD``
# (no ``HERMES_`` prefix); cron/scheduler.py and the CLI still set that.  The
# contextvar takes precedence when set so concurrent gateway tasks don't
# clobber each other's workdir.  Internal debug name matches the file's
# ``HERMES_*`` convention; ``_VAR_MAP`` is keyed by that name too.
_TERMINAL_CWD: ContextVar = ContextVar("HERMES_TERMINAL_CWD", default=_UNSET)

_VAR_MAP = {
    "HERMES_SESSION_PLATFORM": _SESSION_PLATFORM,
    "HERMES_SESSION_CHAT_ID": _SESSION_CHAT_ID,
    "HERMES_SESSION_CHAT_NAME": _SESSION_CHAT_NAME,
    "HERMES_SESSION_THREAD_ID": _SESSION_THREAD_ID,
    "HERMES_SESSION_USER_ID": _SESSION_USER_ID,
    "HERMES_SESSION_USER_NAME": _SESSION_USER_NAME,
    "HERMES_SESSION_KEY": _SESSION_KEY,
    "HERMES_SESSION_ID": _SESSION_ID,
    "HERMES_CRON_AUTO_DELIVER_PLATFORM": _CRON_AUTO_DELIVER_PLATFORM,
    "HERMES_CRON_AUTO_DELIVER_CHAT_ID": _CRON_AUTO_DELIVER_CHAT_ID,
    "HERMES_CRON_AUTO_DELIVER_THREAD_ID": _CRON_AUTO_DELIVER_THREAD_ID,
    "HERMES_TERMINAL_CWD": _TERMINAL_CWD,
}


def set_session_vars(
    platform: str = "",
    chat_id: str = "",
    chat_name: str = "",
    thread_id: str = "",
    user_id: str = "",
    user_name: str = "",
    session_key: str = "",
    terminal_cwd: Optional[str] = None,
) -> list:
    """Set all session context variables and return reset tokens.

    Call ``clear_session_vars(tokens)`` in a ``finally`` block to restore
    the previous values when the handler exits.

    ``terminal_cwd`` defaults to ``None`` (not provided) — in that case the
    ``_TERMINAL_CWD`` contextvar is left at ``_UNSET`` so ``get_terminal_cwd``
    falls back to the legacy ``TERMINAL_CWD`` env var (still set by
    ``cron/scheduler.py``).  Pass an explicit string to scope a per-task cwd.

    Returns a list of ``Token`` objects (one per variable) that can be
    passed to ``clear_session_vars``.
    """
    tokens = [
        _SESSION_PLATFORM.set(platform),
        _SESSION_CHAT_ID.set(chat_id),
        _SESSION_CHAT_NAME.set(chat_name),
        _SESSION_THREAD_ID.set(thread_id),
        _SESSION_USER_ID.set(user_id),
        _SESSION_USER_NAME.set(user_name),
        _SESSION_KEY.set(session_key),
        _TERMINAL_CWD.set(_UNSET if terminal_cwd is None else terminal_cwd),
    ]
    return tokens


def clear_session_vars(tokens: list) -> None:
    """Mark session context variables as explicitly cleared.

    Sets all variables to ``""`` so that ``get_session_env`` returns an empty
    string instead of falling back to (potentially stale) ``os.environ``
    values.  The *tokens* argument is accepted for API compatibility with
    callers that saved the return value of ``set_session_vars``, but the
    actual clearing uses ``var.set("")`` rather than ``var.reset(token)``
    to ensure the "explicitly cleared" state is distinguishable from
    "never set" (which holds the ``_UNSET`` sentinel).
    """
    for var in (
        _SESSION_PLATFORM,
        _SESSION_CHAT_ID,
        _SESSION_CHAT_NAME,
        _SESSION_THREAD_ID,
        _SESSION_USER_ID,
        _SESSION_USER_NAME,
        _SESSION_KEY,
        _TERMINAL_CWD,
    ):
        var.set("")


def get_session_env(name: str, default: str = "") -> str:
    """Read a session context variable by its legacy ``HERMES_SESSION_*`` name.

    Drop-in replacement for ``os.getenv("HERMES_SESSION_*", default)``.

    Resolution order:
    1. Context variable (set by the gateway for concurrency-safe access).
       If the variable was explicitly set (even to ``""``) via
       ``set_session_vars`` or ``clear_session_vars``, that value is
       returned — **no fallback to os.environ**.
    2. ``os.environ`` (only when the context variable was never set in
       this context — i.e. CLI, cron scheduler, and test processes that
       don't use ``set_session_vars`` at all).
    3. *default*
    """
    var = _VAR_MAP.get(name)
    if var is not None:
        value = var.get()
        if value is not _UNSET:
            return value
    # Fall back to os.environ for CLI, cron, and test compatibility
    return os.getenv(name, default)


def set_terminal_cwd(cwd: str):
    """Set the session-scoped terminal cwd and return a reset token.

    Lower-level companion to ``set_session_vars(terminal_cwd=...)`` for
    callers that only need to scope the cwd (e.g. cron job workdir setup).
    """
    return _TERMINAL_CWD.set(cwd)


def reset_terminal_cwd(token) -> None:
    """Restore the previous session-scoped terminal cwd value."""
    _TERMINAL_CWD.reset(token)


def get_terminal_cwd(default: Optional[str] = None) -> str:
    """Backward-compatible accessor for the terminal working directory.

    Prefers the task-local context value when set, otherwise falls back to
    the legacy ``TERMINAL_CWD`` environment variable (still set by
    ``cron/scheduler.py``, the CLI, and ``gateway/run.py``), then to the
    caller-supplied default (or the process cwd if no default is given).

    Resolution order:
    1. ``_TERMINAL_CWD`` contextvar when explicitly set via
       ``set_session_vars(terminal_cwd=...)`` or ``set_terminal_cwd`` —
       that value wins, including the empty-string "explicitly cleared"
       state from ``clear_session_vars`` (which suppresses env-var
       fallback to avoid leaking stale state from a prior gateway session,
       matching the invariant documented on ``get_session_env``).
    2. ``os.environ["TERMINAL_CWD"]`` — consulted only when the contextvar
       is at its ``_UNSET`` sentinel (CLI, cron scheduler, or any
       ``set_session_vars`` call that didn't pass ``terminal_cwd``).
    3. *default* (falling back to ``os.getcwd()`` when ``None``).
    """
    if default is None:
        default = os.getcwd()
    value = _TERMINAL_CWD.get()
    if value is not _UNSET:
        # Explicitly set (or explicitly cleared).  Return as-is when truthy;
        # use the caller default for explicit clears so we don't leak stale
        # os.environ values from a prior session.
        return value if value else default
    return os.environ.get("TERMINAL_CWD", default)

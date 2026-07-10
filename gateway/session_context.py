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

from contextvars import ContextVar
from typing import Any

# Sentinel to distinguish "never set in this context" from "explicitly set to empty".
# When a contextvar holds _UNSET, we fall back to os.environ (CLI/cron compat).
# When it holds "" (after clear_session_vars resets it), we return "" — no fallback.
_UNSET: Any = object()

# Process-level flag: has any code in this process bound a session via
# set_session_vars()? Concurrent multi-session hosts (the messaging gateway, the
# ACP adapter, the API server, the TUI, cron) all do; a pure single-process
# CLI/one-shot that never engages the session-context system does not.
#
# The subprocess-env bridge (tools/environments/local.py) reads this to choose
# its leak policy: when engaged, the ContextVars are authoritative and an _UNSET
# var means "no session bound in THIS task" — so a process-global os.environ
# mirror (written last-writer-wins by whatever concurrent session ran most
# recently) must NOT be inherited into a child process. When never engaged, the
# os.environ fallback is preserved (no concurrency to leak across). Monotonic
# latch — once any host binds a session, the process stays engaged for life.
_session_context_engaged: bool = False


def session_context_engaged() -> bool:
    """True if any session has been bound via set_session_vars in this process.

    See the ``_session_context_engaged`` comment for the leak-policy rationale.
    """
    return _session_context_engaged

# ---------------------------------------------------------------------------
# Per-task session variables
# ---------------------------------------------------------------------------

_SESSION_PLATFORM: ContextVar = ContextVar("HERMES_SESSION_PLATFORM", default=_UNSET)
_SESSION_SOURCE: ContextVar = ContextVar("HERMES_SESSION_SOURCE", default=_UNSET)
_SESSION_CHAT_ID: ContextVar = ContextVar("HERMES_SESSION_CHAT_ID", default=_UNSET)
_SESSION_CHAT_NAME: ContextVar = ContextVar("HERMES_SESSION_CHAT_NAME", default=_UNSET)
_SESSION_THREAD_ID: ContextVar = ContextVar("HERMES_SESSION_THREAD_ID", default=_UNSET)
_SESSION_USER_ID: ContextVar = ContextVar("HERMES_SESSION_USER_ID", default=_UNSET)
_SESSION_USER_NAME: ContextVar = ContextVar("HERMES_SESSION_USER_NAME", default=_UNSET)
_SESSION_KEY: ContextVar = ContextVar("HERMES_SESSION_KEY", default=_UNSET)
_SESSION_ID: ContextVar = ContextVar("HERMES_SESSION_ID", default=_UNSET)
# In-process UI session/window id for multi-session desktop/TUI hosts. This is
# intentionally separate from HERMES_SESSION_ID: the latter is the durable
# conversation/session-db id, while the UI id is the live frontend tab/window
# that commissioned a detached completion. Background completions use it as a
# precise return address so a stale/rotated durable session key cannot be
# consumed by whichever desktop poller wakes first.
_SESSION_UI_SESSION_ID: ContextVar = ContextVar("HERMES_UI_SESSION_ID", default=_UNSET)
# ID of the message that triggered the current turn. Used as a reply anchor
# so background-process notifications stay inside the originating Telegram
# private-chat topic (those lanes route only with thread id + reply anchor).
_SESSION_MESSAGE_ID: ContextVar = ContextVar("HERMES_SESSION_MESSAGE_ID", default=_UNSET)

_SESSION_PROFILE: ContextVar = ContextVar("HERMES_SESSION_PROFILE", default=_UNSET)

# Whether the current session's delivery channel can route an ASYNC completion
# back to the agent AFTER the current turn ends (i.e. wake a fresh turn).
#
# True  — CLI (in-process completion_queue drain) and the real gateway
#         platforms (Telegram/Discord/Slack/...), which hold a persistent
#         outbound channel and run the watcher/drain loops.
# False — stateless request/response adapters (the API server: every route,
#         spec and proprietary, tears down its channel when the turn ends, so
#         a background completion that finishes later has nowhere to go).
#
# Tools that promise async delivery (terminal notify_on_complete /
# watch_patterns, delegate_task background=True) read this via
# ``async_delivery_supported()`` and refuse to hand out a promise the channel
# can't keep — turning a silent no-op into an explicit contract.
#
# Default _UNSET => treated as supported, so CLI (which never sets a platform)
# and any contextvar-unaware path keep working. Stateless adapters opt OUT by
# setting ``supports_async_delivery = False`` on the adapter class; the gateway
# propagates that into this contextvar at session-bind time.
_SESSION_ASYNC_DELIVERY: ContextVar = ContextVar("HERMES_SESSION_ASYNC_DELIVERY", default=_UNSET)

# Cron auto-delivery vars — set per-job in run_job() so concurrent jobs
# don't clobber each other's delivery targets.
_CRON_AUTO_DELIVER_PLATFORM: ContextVar = ContextVar("HERMES_CRON_AUTO_DELIVER_PLATFORM", default=_UNSET)
_CRON_AUTO_DELIVER_CHAT_ID: ContextVar = ContextVar("HERMES_CRON_AUTO_DELIVER_CHAT_ID", default=_UNSET)
_CRON_AUTO_DELIVER_THREAD_ID: ContextVar = ContextVar("HERMES_CRON_AUTO_DELIVER_THREAD_ID", default=_UNSET)

# Cron-session marker — set per-job in run_job() (and re-bound across the
# delegate-task boundary for cron-spawned subagents). Task/thread-isolated so a
# cron job's marker can NEVER bleed into a concurrent interactive gateway turn
# in the same process (the in-gateway cron tick shares the gateway process).
# Previously this was a process-global os.environ flag that latched ON at the
# first cron tick and was never cleared, poisoning send_message origin
# resolution for every later interactive turn (it read cron=True and fell back
# to the home channel). It also gates cron approval auto-mode, so the
# delegate-boundary rebind is required to keep cron subagents deny-gated.
_CRON_SESSION: ContextVar = ContextVar("HERMES_CRON_SESSION", default=_UNSET)

_VAR_MAP = {
    "HERMES_SESSION_PLATFORM": _SESSION_PLATFORM,
    "HERMES_SESSION_SOURCE": _SESSION_SOURCE,
    "HERMES_SESSION_CHAT_ID": _SESSION_CHAT_ID,
    "HERMES_SESSION_CHAT_NAME": _SESSION_CHAT_NAME,
    "HERMES_SESSION_THREAD_ID": _SESSION_THREAD_ID,
    "HERMES_SESSION_USER_ID": _SESSION_USER_ID,
    "HERMES_SESSION_USER_NAME": _SESSION_USER_NAME,
    "HERMES_SESSION_KEY": _SESSION_KEY,
    "HERMES_SESSION_ID": _SESSION_ID,
    "HERMES_UI_SESSION_ID": _SESSION_UI_SESSION_ID,
    "HERMES_SESSION_MESSAGE_ID": _SESSION_MESSAGE_ID,
    "HERMES_SESSION_PROFILE": _SESSION_PROFILE,
    "HERMES_CRON_AUTO_DELIVER_PLATFORM": _CRON_AUTO_DELIVER_PLATFORM,
    "HERMES_CRON_AUTO_DELIVER_CHAT_ID": _CRON_AUTO_DELIVER_CHAT_ID,
    "HERMES_CRON_AUTO_DELIVER_THREAD_ID": _CRON_AUTO_DELIVER_THREAD_ID,
    "HERMES_CRON_SESSION": _CRON_SESSION,
}


def set_current_session_id(session_id: str) -> None:
    """Synchronize ``HERMES_SESSION_ID`` across ContextVar and ``os.environ``.

    Long-lived single-process entrypoints like the CLI can rotate sessions via
    ``/new``, ``/resume``, ``/branch``, or compression splits without
    reconstructing the entire agent. Tools still consult
    ``get_session_env("HERMES_SESSION_ID")`` with an ``os.environ`` fallback,
    so both storage paths must move together when the active session changes.

    🔴 **Gateway-aware (PRD gateway-session-env-leak):** the gateway runs
    concurrent sessions in ONE process, so an unconditional ``os.environ`` write
    here would clobber every other concurrent session's id (the v3-latch bug
    class). Inside the gateway (``_HERMES_GATEWAY=1``) we bind the **contextvar
    only** — the per-turn ``_set_session_env`` already binds ``session_id`` into
    the contextvar, and every gateway-reachable reader is contextvar-first. The
    ``os.environ`` write is kept for **non-gateway single-process** entrypoints
    (CLI rotation, cron-standalone, dispatcher-spawned workers) whose tools read
    the ``os.environ`` fallback and where there is no concurrency to clobber.
    """
    import os

    _SESSION_ID.set(session_id)
    if os.environ.get("_HERMES_GATEWAY") != "1":
        os.environ["HERMES_SESSION_ID"] = session_id


def set_session_vars(
    platform: str = "",
    source: str = "",
    chat_id: str = "",
    chat_name: str = "",
    thread_id: str = "",
    user_id: str = "",
    user_name: str = "",
    session_key: str = "",
    session_id: str = "",
    message_id: str = "",
    profile: str = "",
    cwd: str = "",
    async_delivery: bool = True,
    ui_session_id: str = "",
) -> list:
    """Set all session context variables and return reset tokens.

    Call ``clear_session_vars(tokens)`` in a ``finally`` block when the handler
    exits. Note ``clear_session_vars`` resets every var to ``""`` (to suppress
    the ``os.environ`` fallback) rather than restoring prior values — these
    helpers are not nestable/stack-safe, and the returned tokens are accepted
    only for API compatibility.

    ``cwd`` pins the logical working directory for this context.

    ``async_delivery`` declares whether this session's channel can route a
    background completion back to the agent after the turn ends (see
    ``_SESSION_ASYNC_DELIVERY`` / ``async_delivery_supported``). Stateless
    request/response adapters (the API server) pass ``False``.
    """
    # Mark the session-context machinery engaged for this process. The
    # subprocess-env bridge uses this to switch from "os.environ fallback" to
    # "ContextVar-authoritative, strip on _UNSET" — see session_context_engaged.
    global _session_context_engaged
    _session_context_engaged = True
    tokens = [
        _SESSION_PLATFORM.set(platform),
        _SESSION_SOURCE.set(source),
        _SESSION_CHAT_ID.set(chat_id),
        _SESSION_CHAT_NAME.set(chat_name),
        _SESSION_THREAD_ID.set(thread_id),
        _SESSION_USER_ID.set(user_id),
        _SESSION_USER_NAME.set(user_name),
        _SESSION_KEY.set(session_key),
        _SESSION_ID.set(session_id),
        _SESSION_UI_SESSION_ID.set(ui_session_id),
        _SESSION_MESSAGE_ID.set(message_id),
        _SESSION_PROFILE.set(profile),
        _SESSION_ASYNC_DELIVERY.set(bool(async_delivery)),
    ]
    try:
        from agent.runtime_cwd import set_session_cwd

        set_session_cwd(cwd)
    except Exception:
        pass
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
        _SESSION_SOURCE,
        _SESSION_CHAT_ID,
        _SESSION_CHAT_NAME,
        _SESSION_THREAD_ID,
        _SESSION_USER_ID,
        _SESSION_USER_NAME,
        _SESSION_KEY,
        _SESSION_ID,
        _SESSION_UI_SESSION_ID,
        _SESSION_MESSAGE_ID,
        _SESSION_PROFILE,
    ):
        var.set("")
    # Reset async-delivery capability to the "never set" sentinel rather than a
    # falsy value: a cleared context should fall back to the default-supported
    # behavior (CLI / unaware paths), not be mistaken for an opted-out
    # stateless adapter.
    _SESSION_ASYNC_DELIVERY.set(_UNSET)
    try:
        from agent.runtime_cwd import clear_session_cwd

        clear_session_cwd()
    except Exception:
        pass


def reset_session_vars() -> None:
    """Reset every session context variable to ``_UNSET`` for THIS context.

    Distinct from :func:`clear_session_vars`, which sets the vars to ``""``
    ("explicitly cleared" — suppresses the os.environ fallback and is used when
    a handler *finishes*).  This helper restores the ``_UNSET`` sentinel
    ("never bound in this context"), which is what a freshly-spawned task should
    look like *before* it binds its own session.

    🔴 Why this exists — the cross-session ContextVar inheritance leak.
    Each gateway message is processed in its own ``asyncio`` task, created via
    ``create_task`` (which snapshots the *current* context with
    ``copy_context``).  When message B's task is spawned from a context where a
    concurrent message A had already called :func:`set_session_vars`, B inherits
    A's **set** ContextVars.  Until B calls its own ``set_session_vars`` there is
    a window where any subprocess B spawns (e.g. a tool shelling out) reads
    *A's* ``HERMES_SESSION_*`` identity via the subprocess-env bridge.  The
    bridge's ``_UNSET``-strip guard cannot help: the vars are not ``_UNSET``,
    they are set-to-A.  Calling ``reset_session_vars`` at the top of the
    per-message handler drops the inherited identity so the window strips safe
    (no session) instead of leaking the foreign one; the handler then binds its
    own via ``set_session_vars`` a few steps later.  See
    tests/tools/test_local_env_session_leak.py and
    tests/gateway/test_session_context_inheritance.py.

    Note ``_SESSION_ASYNC_DELIVERY`` lives outside ``_VAR_MAP`` (it is a bool
    capability flag read via :func:`async_delivery_supported`, not a string
    ``HERMES_SESSION_*`` env var read via :func:`get_session_env`), so it is
    reset explicitly below. Without it, a task spawned from a context where a
    sibling adapter bound ``async_delivery=False`` (the stateless API server)
    inherits that ``False`` through the pre-bind window, and
    ``async_delivery_supported`` wrongly reports the new turn's channel as
    unable to route a background completion until ``set_session_vars`` runs.
    """
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    # Reset the async-delivery capability to "never bound here" (_UNSET) for the
    # same inheritance-leak reason as the mapped vars above — see clear_session_vars,
    # which resets this var on the handler-exit path for the symmetric concern.
    _SESSION_ASYNC_DELIVERY.set(_UNSET)
    try:
        from agent.runtime_cwd import clear_session_cwd

        clear_session_cwd()
    except Exception:
        pass


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
    import os

    var = _VAR_MAP.get(name)
    if var is not None:
        value = var.get()
        if value is not _UNSET:
            return value
    # Fall back to os.environ for CLI, cron, and test compatibility
    return os.getenv(name, default)


# ---------------------------------------------------------------------------
# Send-origin context (routing-only) — fixes the subagent send_message leak.
# ---------------------------------------------------------------------------
# A delegate_task child runs in a bare ThreadPoolExecutor worker with an EMPTY
# contextvars context, so the HERMES_SESSION_* vars above are unset inside it.
# A subagent calling send_message/react with a bare target therefore had no
# session origin and silently fell back to the global home channel (the v2
# leak). These dedicated vars carry ONLY the routing origin (platform/chat/
# thread) the send resolver needs — they are read EXCLUSIVELY by
# send_message_tool's origin resolver, never by approval/skills/TTS/cache, so a
# child keeps its own `platform="subagent"` identity for everything else.
#
# CONTEXTVAR-ONLY by design: NO os.environ fallback. An os.environ fallback
# would re-introduce the exact cross-executor leakage we are fixing (a child
# would read the parent process-global). The delegate child-run wrapper binds
# these from the captured parent origin and clears them in a finally.
_SEND_ORIGIN_PLATFORM: ContextVar = ContextVar("HERMES_SEND_ORIGIN_PLATFORM", default=_UNSET)
_SEND_ORIGIN_CHAT_ID: ContextVar = ContextVar("HERMES_SEND_ORIGIN_CHAT_ID", default=_UNSET)
_SEND_ORIGIN_THREAD_ID: ContextVar = ContextVar("HERMES_SEND_ORIGIN_THREAD_ID", default=_UNSET)


def set_send_origin(platform: str, chat_id: str, thread_id: str = "") -> list:
    """Bind the routing-only send origin for the current context (e.g. a
    subagent run). Returns reset tokens; pass to ``clear_send_origin`` in a
    ``finally``. Contextvar-only — never touches os.environ.
    """
    return [
        _SEND_ORIGIN_PLATFORM.set(platform or ""),
        _SEND_ORIGIN_CHAT_ID.set(chat_id or ""),
        _SEND_ORIGIN_THREAD_ID.set(thread_id or ""),
    ]


def clear_send_origin(tokens: list) -> None:
    """Restore the send-origin vars to their prior values (true reset, so this
    IS nestable — a grandchild's clear restores the child's origin, not blank).
    ``tokens`` is the list returned by ``set_send_origin``; ``None``/empty is a
    safe no-op (used when no origin was bound).
    """
    if not tokens:
        return
    for var, token in zip(
        (_SEND_ORIGIN_PLATFORM, _SEND_ORIGIN_CHAT_ID, _SEND_ORIGIN_THREAD_ID),
        tokens,
    ):
        try:
            var.reset(token)
        except Exception:
            # Token from a different context (shouldn't happen — same thread
            # set+clear) — fall back to clearing to empty.
            var.set("")


def get_send_origin() -> tuple:
    """Return ``(platform, chat_id, thread_id)`` for the routing-only send
    origin, or ``("", "", "")`` when unset. Contextvar-only (no os.environ).
    """
    def _v(var):
        val = var.get()
        return "" if val is _UNSET else val

    return (
        _v(_SEND_ORIGIN_PLATFORM),
        _v(_SEND_ORIGIN_CHAT_ID),
        _v(_SEND_ORIGIN_THREAD_ID),
    )


def set_cron_session():
    """Mark the current context (a cron job's run, or a cron-spawned subagent)
    as a cron session. Returns a reset token; pass it to ``clear_cron_session``
    in a ``finally``. Contextvar-only — never touches os.environ (writing the
    process-global env is exactly the latch bug this replaces).

    The set is nestable via ``var.reset(token)`` so a cron subagent that re-binds
    the marker and then clears it restores the parent's value, not blank.
    """
    return _CRON_SESSION.set("1")


def clear_cron_session(token) -> None:
    """Restore the cron-session marker to its prior value. ``token`` is the
    value returned by ``set_cron_session``; ``None`` is a safe no-op (used when
    the marker was not bound — e.g. an interactive parent spawning a child).
    """
    if token is None:
        return
    try:
        _CRON_SESSION.reset(token)
    except Exception:
        # Token from a different context (shouldn't happen — same-thread
        # set+clear). Restore the _UNSET sentinel (NOT "") so get_session_env
        # still falls through to os.environ — setting "" would pin the
        # ContextVar to a non-_UNSET value and silently defeat the I5
        # os.environ back-compat fallback for CLI/standalone/test callers.
        _CRON_SESSION.set(_UNSET)


def is_cron_session() -> bool:
    """True when the current context is a cron job's run (or a cron-spawned
    subagent that re-bound the marker). Reads the ContextVar first, falling back
    to ``os.environ`` for CLI / standalone-scheduler / test compatibility (same
    resolution order as ``get_session_env``).

    This is the single context-aware reader that replaces the raw
    ``env_var_enabled("HERMES_CRON_SESSION")`` calls in approval + send_message.
    Because it goes through ``get_session_env`` (ContextVar -> os.environ ->
    default), a cron job's marker is isolated to its own task/thread lineage and
    can never be read by a concurrent interactive gateway turn.
    """
    from utils import is_truthy_value

    return is_truthy_value(get_session_env("HERMES_CRON_SESSION", ""), default=False)


def async_delivery_supported() -> bool:
    """Whether the current session can deliver a background completion later.

    Returns ``False`` only when the active session was explicitly bound by a
    stateless adapter (the API server) that cannot route a notification back to
    the agent after the turn ends. CLI, cron, and the real gateway platforms —
    and any path that never bound the contextvar — return ``True``.

    Tools that promise async delivery (``terminal`` notify_on_complete /
    watch_patterns, ``delegate_task`` background=True) consult this before
    registering a watcher / dispatching a detached child, so they can refuse a
    promise the channel can't keep instead of silently no-op'ing.
    """
    value = _SESSION_ASYNC_DELIVERY.get()
    if value is _UNSET:
        return True
    return bool(value)

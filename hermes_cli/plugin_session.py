"""The session a plugin hook or tool is running for, read from the turn's task-local context.

Plugin hooks and tools run in the gateway process, whose own cwd is the install root and is shared
by every session (it is never ``os.chdir``-ed). These read the calling session's key, client surface,
recorded working directory and launch policy instead, plus the message event of the running turn.
"""
from __future__ import annotations

import os


def _active_policy():
    from gateway.session_policy import active_policy
    return active_policy()


def session_policy():
    """The launch policy of the session whose turn is running in this context, ``None`` outside one.

    The policy is the session's frozen launch facts: its ``source`` (``cli``/``tui``/``gui``/``acp``),
    ``platform``, the recorded ``cwd``, the resolved ``model`` and the session's ``toolsets``.
    ``session_source()``, ``session_surface()`` and ``session_cwd()`` answer the single fields; read
    the policy for the rest. ``None`` outside a session (the classic CLI, a messaging chat turn, a
    cron run) — the same answer ``session_source()`` gives as an empty string.
    """
    return _active_policy()


def turn_event():
    """The gateway turn's message event for the turn running in this context, ``None`` outside one.

    The event is the platform adapter's own object (a Telegram update, a Discord message): read
    ``user_id`` / ``source`` / ``message_id`` off it and treat every field as untrusted input.
    ``None`` outside a gateway turn (the classic CLI, the TUI) and wherever the platform binds no
    event. Under ``plugins.isolation: host`` the plugin's own process has no turn: it is ``None``
    there.
    """
    from gateway.session_context import current_turn_event
    return current_turn_event()


def session_key() -> str:
    """Route key of the session whose turn is running in this context, ``""`` outside one."""
    from gateway.session_context import get_session_env
    return get_session_env("HERMES_SESSION_KEY", "")


def session_surface() -> str:
    """``cli``/``tui``/``desktop``/``acp`` for a local session's turn, the messaging platform
    (``telegram``, ``discord``, ...) for a chat's turn, ``""`` outside any session."""
    policy = _active_policy()
    if policy is not None:
        return policy.platform
    from gateway.session_context import get_session_env
    return get_session_env("HERMES_SESSION_PLATFORM", "")


def session_client() -> dict | None:
    """The client that created the calling local session, as it declared itself on ``session.create``:
    a copy of ``{"name": ..., "version": ...}`` (``version`` only when sent). ``None`` outside a local
    session, for a messaging chat, and for a client that sent no ``client`` (Ink, Desktop, the CLI and
    ACP today). A label the client chose, not an authentication: never grant anything on it.
    Under ``plugins.isolation: host`` the plugin's own process has no session context; read
    ``info["client"]`` in a prompt section there."""
    from gateway.session_client import policy_client
    return policy_client(_active_policy())


def turn_client() -> dict | None:
    """The client that submitted the turn running in this context, as its connection announced
    itself (``client.capabilities``): a copy of ``{"name": ..., "version": ..., "kind": ...}``
    (``version`` / ``kind`` only when sent; a kind this Hermes does not know is left out). ``None``
    outside a turn, in a delegated child (it runs in its parent turn's context, not under an admission
    of its own), for a client that announced nothing, and for a messaging, cron or other automation
    turn. Answers only inside the gateway process: under ``plugins.isolation: host`` it is always
    ``None`` in the plugin's own process, where ``ctx.turn_client()`` asks the gateway instead. A label
    the client chose, not an authentication: never grant anything on it."""
    if _in_delegated_child():
        return None
    from gateway.session_surface import surface_turn_client
    return surface_turn_client()


def turn_origin() -> str | None:
    """The Hermes producer of the turn running in this context, the pair of ``turn_client()``, named
    only where Hermes knows it: ``"process_notification"`` for a turn a background process's
    completion, watch match or liveness tick started, ``"delegation_result"`` for an async
    delegation's result re-entering its parent. ``None`` for every other turn, whoever started it:
    a message typed in a client or sent on a messaging chat, an API request, a Bot Chat delivery or
    ``hermes peer dm``, a restart's resume of an interrupted turn, a goal continuation, cron, Kanban
    and other automation wakes, a message a plugin injected, and a steer or interrupt text run as the
    next turn. ``None`` too outside a turn, in a delegated child, in a detached fork (``fork_origin()``
    non-empty: a plugin fork, a side question, a background review runs its own prompt in a copy of the
    turn's context) and in a managed worker. New values may be added: compare against the ones you
    handle. A person's input that joins the running turn (an accepted steer or redirect, from a client's
    Steer, ``/steer``, ``session.steer``/``session.redirect`` or a busy chat message) makes it ``None``
    for the rest of that turn: the reply now answers the person too. Answers only inside the gateway
    process: under ``plugins.isolation: host`` it is always ``None`` in the plugin's own process, where
    ``ctx.turn_origin()`` asks the gateway instead."""
    if _in_delegated_child() or fork_origin():
        return None
    from gateway.session_context import bound_turn_origin
    return bound_turn_origin()


def session_viewers() -> list[dict]:
    """The clients attached to the calling session right now, one dict per connection:
    ``{"name", "kind"}`` (``None`` when the connection announced nothing), ``"idle_seconds"`` since the
    user's last input there (``None`` when none was ever seen for this session), ``"basis"`` (``"system"``:
    the device's idle clock, ``"page"``: input inside the Hermes window, ``"input"``: only what this
    gateway saw, ``None`` with no input) and ``"can_alert"`` (whether that client can get the user's
    attention itself; ``None`` while no report of that client counts for this session, and a report
    that leaves it out says ``True``). The calling session is the one this context bound: a
    turn's, or the one a plugin's own slash command was typed in (no turn runs there, so
    ``turn_client()`` is ``None``). ``[]`` when none is bound (a context that bound only a route, such
    as a session open or a built-in slash command), in a delegated child, in a managed worker, and
    when the session is not live. Answers only inside the gateway process: under
    ``plugins.isolation: host`` it is always ``[]`` in the plugin's own process (no gateway there),
    where ``ctx.session_viewers()`` asks the gateway instead. Labels and reports the clients chose,
    not an authentication: never grant anything on them."""
    from gateway.session_context import bound_session_env, bound_session_key
    # Only what this context bound, never ``os.environ``: the live session the turn runner itself
    # binds (``TurnRunner._approval_owner``) and a plugin command's scope binds the same way: a local
    # session's chat id is its canonical id, any other runs under its logical owner. The live session
    # must also answer to the bound route, so another session's id can never be borrowed.
    route = bound_session_key()
    local = bound_session_env("HERMES_SESSION_PLATFORM") == "local"
    session_id = bound_session_env("HERMES_SESSION_CHAT_ID" if local else "HERMES_SESSION_ID")
    authority = _turn_authority() if route and session_id and not _in_delegated_child() else None
    if authority is None:
        return []
    if not local:
        session_id = authority.logical_owner(session_id)
    live = authority.sessions.get(session_id)
    if live is None or live.route != route:
        return []
    from gateway.session_presence import viewers
    return viewers(authority, session_id)


def session_source() -> str:
    """The label the calling session was created with: its launch policy's ``source``. ``cli``,
    ``tui``, ``gui`` or ``acp`` for a chat opened in that client, ``tool`` for a run a third-party
    integration launched (``hermes chat --source tool``), ``oneshot`` for a finite ``chat -q`` /
    ``-z`` run; any other non-empty value is a label Hermes gives a session it starts itself. It is a
    launch label, not proof that a person is waiting. ``""`` says only that no launch policy is
    bound: outside a session, for a person's messaging chat turn and for a cron run
    (``current_cron_execution`` names that one). A delegated child runs in its parent turn's context
    and reads the parent session's label. Never read from ``os.environ``."""
    policy = _active_policy()
    return policy.source if policy is not None else ""


def in_delegated_child(*, include_subprocesses: bool = False) -> bool:
    """``True`` while the calling code runs for a ``delegate_task`` child. With
    ``include_subprocesses=True``, also ``True`` in a process a delegated child started (Hermes marks
    that process's environment)."""
    from agent.delegation_context import is_delegated_child_context, is_delegated_child_process_context
    return is_delegated_child_process_context() if include_subprocesses else is_delegated_child_context()


def fork_origin() -> str:
    """The running detached fork's label: ``"background_review"``, ``"side_question"``, or
    ``"plugin_fork:<plugin id>"``. ``""`` outside a detached fork, regardless of which skill or
    memory write rules apply. Use ``ctx.fork_origin()`` from a hosted plugin."""
    from agent.turn_facade import detached_fork_origin
    return detached_fork_origin()


def _in_delegated_child():
    """A delegated child runs in a copy of its parent turn's context (stamp and session vars
    included), yet no admission of its own executes there: the turn's facts are not the child's."""
    from agent.delegation_context import is_delegated_child_context
    return is_delegated_child_context()


def _turn_authority():
    """The session authority of the profile this turn runs under, the lookup ``TurnRunner`` binds its
    owner with: the active runtime scope's, never the launch profile's for a routed one. ``None``
    with no gateway in this process (CLI, managed worker, plugin host)."""
    from gateway.run import _gateway_runner_ref
    from gateway.session_authorities import active_authority
    runner = _gateway_runner_ref()
    return active_authority(runner) if runner is not None else None


def session_cwd() -> str:
    """The calling session's recorded working directory (a local session's workspace, else the
    profile's terminal cwd), ``""`` when it has none. Never the gateway process's own cwd.
    In ingress hooks (``pre_gateway_dispatch``), which run before any session, it is the profile's
    terminal cwd, not the workspace of the session the message will reach."""
    from agent.runtime_cwd import scope_terminal_cwd, scoped_session_cwd
    policy = _active_policy()
    return scoped_session_cwd() or (policy.cwd if policy is not None else "") or scope_terminal_cwd().strip()


def context_surface(manager) -> str:
    """``session_surface()`` for a plugin context: the classic CLI process itself is ``cli``."""
    return "cli" if manager._cli_ref is not None else session_surface()


def context_cwd(manager) -> str:
    """``session_cwd()`` for a plugin context: the classic CLI process keeps its launch directory."""
    return os.getcwd() if manager._cli_ref is not None else session_cwd()

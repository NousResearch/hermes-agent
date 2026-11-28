"""When a viewer's close ends the session's async delegations: at once, or after a grace.

``session.detach {end_reason}`` from the last viewer ends the session's own background delegations
(#134889 LC-7). Two kinds of reason reach it:

* ``cli_close`` / ``new_session``: the classic CLI's own ``/exit`` and ``/new``. Deliberate; they end
  the work at once.
* ``tui_close`` / ``tui_shutdown``: the Ink TUI's view teardown. It sends them for ``/quit`` and
  ``/new``, and equally when its transport fails and it tears itself down (a stalled gateway loop, a
  dropped websocket, a reconnect). The owner cannot tell these apart, and the session carries on
  either way: a viewer comes back, the user keeps chatting, and the hours of subagent work that was
  interrupted at the first sign of a dropped socket are gone.

For the second kind the interrupt waits ``sessions.viewer_close_delegation_grace_s`` (default 300 s,
0 = at once). Any viewer attaching to the session in that time cancels it; an exit nobody takes back
still ends the work when the grace runs out. The close itself (row, memory commit, finalize hooks)
is not delayed: only the delegations' fate is.
"""
import asyncio
import logging

from gateway.session_history_mount import is_history_mount

logger = logging.getLogger(__name__)

# The TUI's teardown reasons: they also fire for a viewer that merely lost its transport.
VIEWER_TEARDOWN_REASONS = frozenset({'tui_close', 'tui_shutdown'})
DEFAULT_GRACE_S = 300.0


def delegation_end_grace(authority, reason):
    """Blocking: seconds to wait before a close with *reason* ends the session's delegations (0 = now)."""
    if reason not in VIEWER_TEARDOWN_REASONS:
        return 0.0
    from gateway.session_viewer_close import _section, _seconds, authority_persistence
    from gateway.session_local import profile_env_scope
    from hermes_cli.config_effective import load_user_config_effective
    home = authority_persistence(authority).scope.profile_dir
    with profile_env_scope(home):
        config = load_user_config_effective(home / 'config.yaml')
    return _seconds(_section(config, 'sessions').get('viewer_close_delegation_grace_s'), DEFAULT_GRACE_S)


def end_delegations(authority, session_id, route, parents, reason):
    """Refuse new background work for the session and interrupt what it has running (main's exit/``/new``)."""
    from tools.async_delegation import interrupt_for_session, refuse_for_session
    refuse_for_session((authority.profile_id, session_id), session_key=route, parent_session_ids=parents)
    for parent in parents:
        interrupt_for_session(session_key=route, parent_session_id=parent, reason=f'viewer_close:{reason}')


def defer_delegation_end(authority, session_id, route, parents, reason, grace):
    """End the delegations after *grace* seconds unless a viewer attaches to the session first."""
    from gateway.session_viewer_close import _call_later, _stopping
    cancel_deferred_delegation_end(authority, session_id)

    def fire():
        authority.delegation_ends.pop(session_id, None)
        live = authority.sessions.get(session_id)
        if _stopping(authority) or (live is not None and (live.subscribers or is_history_mount(live))):
            return  # a viewer is back (or the owner is stopping, which ends the work with the process)
        logger.info('Viewer close %s on %s was not taken back within %.0fs: ending its delegations',
                    reason, session_id, grace)
        end_delegations(authority, session_id, route, parents, reason)
    authority.delegation_ends[session_id] = _call_later(asyncio.get_running_loop(), grace, fire)


def cancel_deferred_delegation_end(authority, session_id):
    handle = authority.delegation_ends.pop(session_id, None)
    if handle is not None:
        handle.cancel()

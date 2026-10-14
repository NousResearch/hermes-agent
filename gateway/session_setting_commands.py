"""The messaging setting handlers (``/reasoning`` ``/busy`` ``/verbose`` ``/fast`` ...) on a LOCAL session.

``slash.exec`` runs the same ``_handle_<name>_command`` a messaging chat runs. A local session's
next turn reads its settings from its live session and frozen policy, not from the runner's
per-platform snapshot, so a handler that changes a setting also applies it here: one behaviour
for every surface, held by the gateway.
"""


def local_live(runner, source):
    """The ``LiveSession`` of the local session ``source`` belongs to, else None (messaging chats)."""
    from gateway.config import Platform
    if source is None or getattr(source, 'platform', None) != Platform.LOCAL:
        return None
    from gateway.session_authorities import all_authorities
    for authority in all_authorities(runner):
        live = getattr(authority, 'sessions', {}).get(source.chat_id)
        if live is not None and live.source.platform == Platform.LOCAL:
            return live
    return None


def local_policy(runner, source):
    """The frozen policy of a local session, else None."""
    if local_live(runner, source) is None:
        return None
    from gateway.session_policy import policy_for_source
    return policy_for_source(runner, source)


def session_busy_mode(runner, live):
    """The busy mode a local session's next input runs under (what ``config.get busy`` reports)."""
    value = getattr(live, 'busy_input_mode', None)
    if value in ('interrupt', 'steer', 'queue'):
        return value
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(runner, live.source)
    # Main's TUI defaulted to ``queue`` (``ui-tui/src/app/useConfigSync.ts`` TUI_BUSY_DEFAULT): in a
    # full-screen composer the next prompt is authored while the agent streams, and an unintended
    # interrupt loses work. Only an explicit ``display.busy_input_mode`` overrides it; every other
    # surface keeps the framework default ``interrupt``.
    fallback = 'queue' if policy is not None and getattr(policy, 'source', None) == 'tui' else 'interrupt'
    if policy is None:
        return runner._effective_busy_input_mode(live.source)
    value = (policy.config().get('display') or {}).get('busy_input_mode') or fallback
    return value if value in ('interrupt', 'steer', 'queue') else fallback


def local_live_for_route(runner, session_key):
    """The local ``LiveSession`` whose route is ``session_key``, else None."""
    if not session_key:
        return None
    from gateway.config import Platform
    from gateway.session_authorities import all_authorities
    for authority in all_authorities(runner):
        for live in list(getattr(authority, 'sessions', {}).values()):
            if live.route == session_key and live.source.platform == Platform.LOCAL:
                return live
    return None

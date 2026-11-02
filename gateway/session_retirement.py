"""Idle conversations leave the authority's memory, as a restart leaves them.

A ``LiveSession`` (event stream, replay ring key, locks, pending controls) is only needed while a
viewer watches the conversation or input runs in it. A local conversation its last viewer closed,
and an API conversation between requests, otherwise stayed in ``authority.sessions`` until the
process exited. Every entry point already handles an absent session (that is the state after a
restart): attach and admission restore it from its receipt, an API request rebinds it, and input
committed for it is restored by ``_schedule``/``_publish_pending`` before the FIFO runs it.
"""
import asyncio

from gateway.config import Platform

_RETIRABLE = frozenset({Platform.LOCAL, Platform.API_SERVER})


def _pinned(authority, session_id, live):
    """In-memory state that still needs this exact ``LiveSession``."""
    from gateway.session_history_mount import is_history_mount
    task = live.task
    return (authority.sessions.get(session_id) is not live
            or live.source is None or live.source.platform not in _RETIRABLE
            or live.subscribers or live.event_stream.observers or live.rescan or live.settling
            or live.detached or is_history_mount(live)
            or live.mutation_lock.locked() or live.claim_gate.locked()
            or (task is not None and not task.done() and task is not asyncio.current_task())
            or session_id in authority.executing or session_id in authority.pending_closes
            or session_id in authority.orphan_closes)


def retire_idle(authority, session_id, live, target):
    """Drop *live* when nothing pins it. Called with no await since the last check before it, and
    with none inside: the FIFO read below and the removal are one step on the loop. Input committed
    before that read shows as a queued row (or a worker as registered) and keeps the session; input
    committing after it is followed by a loop-side ``_schedule``/``_publish_pending``, which
    restores the conversation (``restore_retired``) before anything indexes it."""
    if _pinned(authority, session_id, live):
        return False
    from gateway.session_viewer_close import _busy
    if _busy(authority, session_id, target):
        return False
    del authority.sessions[session_id]
    authority.launch_sources.pop(session_id, None)  # a cache: re-read from the row on demand
    if live.source.platform == Platform.LOCAL:
        authority.retired_local.add(session_id)
    return True


def retire_closed(authority, session_id, live, target, idle_stamp):
    """A closed local conversation nobody views and nothing runs in. A turn claimed since the close
    found the FIFO idle (*idle_stamp* replaced) reopened its row: it stays. A close whose retirement a
    running drain pinned is retired by that drain's idle exit (``retire_at_idle``)."""
    live.closed_stamp = (idle_stamp,)
    if live.event_stream.execution is not idle_stamp:
        return False
    return retire_idle(authority, session_id, live, target)


def retire_at_idle(authority, ref, live):
    """The drain's idle exit: an API conversation between requests, or a closed local one whose
    close ran while this drain was alive."""
    platform = live.source.platform if live.source is not None else None
    if platform == Platform.API_SERVER:
        return retire_idle(authority, ref.session_id, live, authority.physical_target(ref))
    stamp = live.closed_stamp
    if platform == Platform.LOCAL and stamp is not None and stamp[0] is live.event_stream.execution:
        from gateway.session_viewer_close import _physical_target
        return retire_idle(authority, ref.session_id, live, _physical_target(authority, ref.session_id))
    return False


def restore_retired(authority, session_id):
    """Bring back a conversation ``retire_idle`` let go, from its receipt. A local one keeps its
    frozen policy on the local adapter until it is deleted; an API one keeps its binding receipt.
    Anything else (deleted, never bound) stays absent, as before retirement existed."""
    if session_id in authority.sessions:
        return
    if session_id in authority.retired_local:
        from gateway.session_local_recovery import restore_local_session
        restore_local_session(authority, session_id)
        return
    from gateway.session_api import restore_api_session
    from hermes_state_runtime import RuntimeStoreError
    try:
        restore_api_session(authority, session_id)
    except RuntimeStoreError:
        pass  # no binding: the caller's lookup fails exactly as it did before retirement


def forget_retired(authority, session_id):
    """A deleted conversation retirement let go: its admission source and frozen policy on the local
    adapter, and its cached agent (``forget_live_session`` for one still in memory)."""
    authority.retired_local.discard(session_id)
    from gateway.session_local_recovery import local_adapter_map
    local = local_adapter_map(authority).get(Platform.LOCAL)
    saved = local.sources.get(session_id) if local is not None else None
    if saved is None:
        return
    route = authority.runner.session_store._generate_session_key(saved[0])
    evict = getattr(authority.runner, '_evict_cached_agent', None)
    if callable(evict):
        evict(route)
    local.forget(saved[0], route)

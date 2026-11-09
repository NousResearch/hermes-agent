"""A completion for a local session no viewer is watching is written into it as a note (#134889 LC-7).

Main's CLI and TUI hosted async delegations and background processes in their own process, so once
the user left nothing started a turn for them. Under the gateway the owner outlives its viewers. The
completion is admitted like any producer input. Producers still coalesce a batch into one admission
under its primary identity, the siblings' identities in its descriptor: a lookup by any of them finds
it (``completion_admission``), results already admitted are left out of a new batch, and the one
outcome, note or turn, covers the whole batch. When that admission comes up for execution with no
viewer attached (the authority's own subscriber records) the executor records it in the session's
history, where the next turn sees it, and settles the admission without a model call. Queue order, restart survival and
exactly-once are the admission contract's. With a viewer attached it runs as the turn it always was,
except a local cron result, which is always a note; messaging, Kanban, ``/loop`` and heartbeat wakes
never reach here.
"""
import json
from functools import partial

# ``_completion_delivery_identity`` kinds: an async delegation's result, a background process's exit
# and its watch-pattern notices.
_NOTED_PRODUCERS = frozenset({'async_delegation', 'completion', 'watch_match', 'watch_disabled'})
# A cron result for a local chat (``session_cron.deliver_to_local_chat``) is finished text: a note
# with or without a viewer, since a turn over it would only spend a model call (#143547 CRON-11).
_ALWAYS_NOTED = frozenset({'cron'})


def _producer(identity):
    try:
        return json.loads(identity)[0]
    except (ValueError, TypeError, IndexError, KeyError):
        return None


def noted_completion(row):
    """*row* is a local completion admission: a note when it runs with no viewer attached."""
    descriptor = (row or {}).get('payload', {}).get('local_automation_v1')
    return descriptor is not None and _producer(descriptor['identity']) in _NOTED_PRODUCERS | _ALWAYS_NOTED


async def reopen_for_turn(authority, ref):
    """A turn (never a note) makes *ref*'s finalized row live again: a local session's receipt tip, or
    the row of a conversation its chat moved past (gateway/session_detached.py), which the chat's /new
    ended."""
    from gateway.session_runtime_workers import tracked_write
    live = authority.sessions.get(ref.session_id)
    if live is not None and live.detached:
        from gateway.session_api import reopen_api_session as reopen
    else:
        from gateway.session_local_recovery import reopen_local_session as reopen
    await tracked_write(authority, partial(reopen, authority, ref))


async def note_if_unviewed(authority, ref, row):
    """The claimed *row*'s response when it was written as a note instead of running, else None.
    It is the FIFO head, so the note lands after every input committed before it and before every
    input committed after; the viewer check and the decision share one loop step."""
    if not noted_completion(row):
        return None
    from gateway.session_runtime_workers import tracked_write
    live = authority.sessions.get(ref.session_id)
    descriptor = row['payload']['local_automation_v1']
    if live is None or (live.subscribers and _producer(descriptor['identity']) not in _ALWAYS_NOTED):
        # The claim left a closed row closed for a note; the turn reopens it as any first turn does.
        await reopen_for_turn(authority, ref)
        return None
    from gateway.session_viewer_close import _physical_target
    metadata = {'completion_note': sorted(set(descriptor.get('identities') or ()) | {descriptor['identity']})}
    metadata.update({key: descriptor[key] for key in ('display_text', 'notification_category') if key in descriptor})
    target = _physical_target(authority, ref.session_id)
    # Keyed by its admission: a claim is never replayed, and this keeps a write retry exact.
    await tracked_write(authority, partial(
        authority.db.append_completion_note, target, row['admission_id'], row['payload']['text'],
        descriptor.get('display_kind', 'internal_notification'), metadata))
    refresh = getattr(authority.runner, '_refresh_agent_cache_message_count', None)
    if refresh is not None:  # the owner's own row: a cached agent stays, as after a turn's writes
        await refresh(live.route, target)
    authority.pending_results[row['admission_id']] = {'result': {'final_response': '', 'messages': []}, 'usage': {}}
    return ''

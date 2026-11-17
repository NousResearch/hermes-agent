"""ACP v0.9 has no per-session destroy: the last viewer leaving an idle ACP session ends it (#118216).

The in-process adapter stamped ``ended_at`` at its stdio shutdown; under the gateway the ACP process
is only a viewer, so the owner stamps it when the last subscription goes or the gateway stops. Without
a writer, source='acp' rows stay open forever and the ended-session guard keeps prune/archive away.
A later ``session/load`` + prompt reopens the row (``session_local_recovery.reopen_local_session``).
"""
import logging

from hermes_state_local import end_idle_local_session, local_receipt
from hermes_state_runtime import RuntimeStoreError

ACP_END_REASON = 'acp_disconnect'


def end_idle_acp_session(authority, session_id):
    """End *session_id* iff it is an ACP-created local session with no viewer and an idle FIFO."""
    live = authority.sessions.get(session_id)
    if live is not None and live.subscribers:
        return False
    try:
        receipt = local_receipt(authority.db, session_id)
    except RuntimeStoreError:
        return False  # not a local session: no ACP policy to end
    if receipt.get('policy', {}).get('source') != 'acp':
        return False
    try:
        return bool(end_idle_local_session(authority.db, epoch=authority.epoch, session_id=session_id,
                                           target_id=receipt['entry']['session_id'], reason=ACP_END_REASON))
    except (RuntimeStoreError, KeyError, TypeError):
        logging.getLogger(__name__).warning('ACP session %s not ended', session_id, exc_info=True)
        return False


def end_idle_viewer_session(authority, session_id, reason):
    """``session.detach {end_reason}`` from the last viewer: end *session_id*'s idle local row with the
    viewer's own reason (main's classic CLI ``end_session(sid, 'cli_close')`` and its switch reasons).
    ACP rows keep their own reason; a busy FIFO keeps the row open, as for ACP."""
    live = authority.sessions.get(session_id)
    if live is not None and live.subscribers:
        return False
    try:
        receipt = local_receipt(authority.db, session_id)
    except RuntimeStoreError:
        return False  # not a local session: nothing this verb may end
    if receipt.get('policy', {}).get('source') == 'acp':
        return end_idle_acp_session(authority, session_id)
    try:
        return bool(end_idle_local_session(authority.db, epoch=authority.epoch, session_id=session_id,
                                           target_id=receipt['entry']['session_id'], reason=reason))
    except (RuntimeStoreError, KeyError, TypeError):
        logging.getLogger(__name__).warning('local session %s not ended', session_id, exc_info=True)
        return False


async def reopen_if_reattached(authority, session_id, ended):
    """On the owner loop after the close committed: a viewer attached meanwhile, so the session
    it watches is live again (the stamp would make active work prune/archive eligible)."""
    live = authority.sessions.get(session_id)
    if ended and live is not None and live.subscribers:
        import asyncio
        target = local_receipt(authority.db, session_id)['entry']['session_id']
        await asyncio.to_thread(authority.db.reopen_session, target)
    return ended


def acp_candidate_ids(db):
    """Ids that can hold an ACP policy receipt, read from the indexed ``sessions`` table.

    A receipt with ``policy.source='acp'`` creates its row as ``source='acp'``, and a peer refresh
    relabels it ``source='local'`` with ``created_source='acp'`` (``session_policy.STORED_AS_LOCAL``).
    A candidate is only a lookup key: ``end_idle_acp_session`` still judges the receipt's own policy.
    """
    with db._read_ctx() as conn:
        return [row[0] for row in conn.execute(
            "SELECT id FROM sessions WHERE source = 'acp' OR (source = 'local' AND created_source = 'acp')")]


def end_idle_acp_sessions(authority):
    """Gateway shutdown: every ACP session nobody is viewing, attached this run or not.

    Costs one receipt read per ACP session, not one per local session ever created: the policy
    receipts of every other surface are never parsed (``acp_candidate_ids``)."""
    return sum(end_idle_acp_session(authority, sid) for sid in acp_candidate_ids(authority.db))

"""ACP v0.9 has no per-session destroy: the last viewer leaving an idle ACP session ends it (#118216).

The in-process adapter stamped ``ended_at`` at its stdio shutdown; under the gateway the ACP process
is only a viewer, so the owner stamps it when the last subscription goes or the gateway stops. Without
a writer, source='acp' rows stay open forever and the ended-session guard keeps prune/archive away.
A later ``session/load`` + prompt reopens the row (``session_local_recovery.reopen_local_session``).
"""
import logging

from hermes_state_local import POLICY_PREFIX, end_idle_local_session, local_receipt
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


def end_idle_acp_sessions(authority):
    """Gateway shutdown: every ACP session nobody is viewing, attached this run or not."""
    with authority.db._read_ctx() as conn:
        ids = [row[0][len(POLICY_PREFIX):] for row in conn.execute(
            'SELECT key FROM state_meta WHERE key LIKE ?', (POLICY_PREFIX + '%',))]
    return sum(end_idle_acp_session(authority, sid) for sid in ids)

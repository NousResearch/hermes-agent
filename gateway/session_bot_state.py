"""A Bot Chat's live state, handed to the teammate whose ``message_agent`` delivery it answers.

A sender that only learns "queued" cannot tell a busy teammate from one parked on its user's
approval or clarify card; it waits out the bound and reports a silent timeout. The deliver answer
(and every same-id replay the waiting runner makes) carries this state instead. Read on the owner
loop, never persisted: a receipt holds outcomes, not a snapshot that is stale a second later.
"""
from gateway.session_contract import SessionRef

# Prompt kinds a running turn parks on, in the order a sender should hear about them.
_WAITING_KINDS = ('approval', 'clarify')


def bot_chat_state(authority, session_id):
    """``paused`` (the owner admits nothing new: a drain for restart/update or retirement),
    ``idle``, ``working``, or ``waiting_approval`` / ``waiting_clarify`` while the running turn
    waits on its user."""
    if authority.retiring or authority.runner._draining:
        return 'paused'
    live = authority.sessions.get(session_id)
    if live is None:
        return 'idle'
    handle = authority._handle(SessionRef(authority.profile_id, session_id))
    if handle.execution_state != 'running':
        return 'idle'
    kinds = {prompt['kind'] for prompt in live.controls.snapshot(session_id, handle.execution_generation)}
    return next((f'waiting_{kind}' for kind in _WAITING_KINDS if kind in kinds), 'working')

"""Private per-admission binding for a user row already written by the owner."""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from persistence.core.identity import MessageId, SessionId

if TYPE_CHECKING:
    from agent.turn_context import DurableUserInput

_bound: ContextVar[tuple[str, tuple['DurableUserInput', str]] | None] = ContextVar('gateway_stored_input', default=None)


def decode_stored_input(proof, session_id):
    """Validate the owner-to-executor handoff, never a client payload."""
    from agent.turn_context import DurableUserInput
    if (not isinstance(proof, dict) or set(proof) != {'session_id', 'row_id', 'message_uid', 'text'}
            or not isinstance(proof['session_id'], str) or not proof['session_id']
            or proof['session_id'] != session_id or type(proof['row_id']) is not int or proof['row_id'] <= 0
            or not isinstance(proof['message_uid'], str) or not proof['message_uid']
            or not isinstance(proof['text'], str)):
        raise ValueError('invalid stored user input proof')
    return DurableUserInput(SessionId(proof['session_id']), MessageId(proof['row_id']), proof['message_uid']), proof['text']


@contextmanager
def stored_input_scope(admission_id, session_id, proof):
    value = (admission_id, decode_stored_input(proof, session_id)) if proof is not None else None
    token = _bound.set(value)
    try:
        yield
    finally:
        _bound.reset(token)


def stored_input(admission_id):
    value = _bound.get()
    return value[1] if value is not None and value[0] == admission_id else None

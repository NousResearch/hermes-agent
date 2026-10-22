"""Reconcile ordinary admitted turns after a viewer missed their stream."""
import logging
import time

from hermes_cli.gateway_client import GatewayClientError

logger = logging.getLogger(__name__)
REATTACH_RETRY_SECONDS = 5


def finish_viewed_turn(state, admission, result) -> None:
    if not admission:
        return
    if result is None or result.get('failure_reason') == 'gateway_connection':
        if admission in state.queued_inputs:
            state.missed.add(admission)
        return
    state.painted_admissions.add(admission)
    state.queued_inputs.pop(admission, None)
    state.missed.discard(admission)


def remember_input(state, admission, public_input) -> None:
    if not admission or admission in state.painted_admissions:
        return
    if isinstance(public_input, dict) and isinstance(public_input.get('text'), str) and public_input['text']:
        state.queued_inputs.setdefault(admission, public_input['text'])


def recover_history(store, state) -> None:
    if not state.history_needed or state.admission_cursor is None:
        return
    page = store.call('prompt.history', session_id=state.session_id, after_sequence=state.admission_cursor)
    for entry in page['admissions']:
        receipt = entry['receipt']
        remember_input(state, receipt['admission_id'], entry['input'])
        if receipt['status'] != 'queued' and receipt['admission_id'] in state.queued_inputs:
            state.missed.add(receipt['admission_id'])
    state.admission_cursor = page['last_sequence']
    state.history_needed = page['has_more']


def reattach_idle(store, state) -> bool:
    if time.monotonic() < state.reattach_after:
        return False
    try:
        store.connection.reattach(state.session_id)
    except GatewayClientError as exc:
        state.reattach_after = time.monotonic() + REATTACH_RETRY_SECONDS
        logger.warning('Idle session reconnect failed: %s', exc)
        return False
    state.reattach_after = 0
    return True

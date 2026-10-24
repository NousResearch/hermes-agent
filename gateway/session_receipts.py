"""Durable admission projections; reading a receipt never schedules input."""
from dataclasses import asdict

from pydantic import ValidationError

from gateway.session_automation import admission_display
from hermes_state_admission_history import admission_history
from hermes_state_runtime import RuntimeStoreError
from tui_gateway.contracts.canonical import PromptHistoryParams


async def history(connection, ref, params):
    authority = connection.authority
    authority.authorize(connection.actor, ref, 'session:read')
    try:
        query = PromptHistoryParams.model_validate(params)
    except ValidationError as exc:
        raise RuntimeStoreError('invalid_params') from exc
    rows = admission_history(authority.db, ref.session_id, query.after_sequence, query.limit + 1)
    page = rows[:query.limit]
    live = authority.sessions[ref.session_id]
    with live.event_stream.lock:
        receipts = [{'receipt': asdict(authority._receipt(live.settling.get(row['admission_id'], row))),
                     'input': admission_display(row['payload']).get('input')} for row in page]
    return {'admissions': receipts, 'last_sequence': page[-1]['seq'] if page else query.after_sequence,
            'has_more': len(rows) > query.limit}

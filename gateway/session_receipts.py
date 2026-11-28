"""Durable admission projections; reading a receipt never schedules input."""
from dataclasses import asdict

from pydantic import ValidationError

from gateway.session_authority import admission_row
from gateway.session_automation import admission_display
from gateway.session_persistence import authority_persistence
from persistence.core.errors import AuthorityRefusal
from tui_gateway.contracts.canonical import PromptHistoryParams


async def history(connection, ref, params):
    authority = connection.authority
    authority.authorize(connection.actor, ref, 'session:read')
    try:
        query = PromptHistoryParams.model_validate(params)
    except ValidationError as exc:
        raise AuthorityRefusal('invalid_params') from exc
    rows = [admission_row(record) for record in authority_persistence(authority).admissions.history(
        session_id=ref.session_id, after_seq=query.after_sequence, limit=query.limit + 1)]
    page = rows[:query.limit]
    live = authority.sessions[ref.session_id]
    with live.event_stream.lock:
        receipts = [{'receipt': asdict(authority._receipt(live.settling.get(row['admission_id'], row))),
                     'input': admission_display(row['payload']).get('input')} for row in page]
    return {'admissions': receipts, 'last_sequence': page[-1]['seq'] if page else query.after_sequence,
            'has_more': len(rows) > query.limit}

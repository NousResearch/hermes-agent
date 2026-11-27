"""Shared queue selection for owner preflight and the atomic ledger claim."""

import json


def preferred_session_input(rows):
    """Select a queue head without bypassing preflight or changing durable identity.

    Only an authenticated operator envelope counts as a human, not every principal
    outside automation (workers/cron/control submissions are not operator turns).
    Only Kanban wakes can yield; other producers keep their relative FIFO order.
    """
    queued = [row for row in rows if row['status'] == 'queued']
    human_seq = max((r['seq'] for r in queued if is_operator_input(r)), default=0)
    return next((r for r in queued if not (r['seq'] < human_seq and is_kanban_input(r))),
                queued[0] if queued else None)


def is_operator_input(row):
    payload = row['payload']
    if row['principal_id'].startswith(('automation:', 'worker:', 'cron:')):
        return False
    if payload.get('local_automation_v1') or payload.get('kanban_card_v1') or payload.get('unattended'):
        return False
    native = payload.get('native_text_v1')
    if native is not None:
        return not native.get('automation')
    # Plain authenticated session.submit (CLI/API), including a local owner
    # whose route has not yet acquired a local_operator_v1 binding.
    return 'text' in payload


def is_kanban_input(row):
    try:
        return (row['principal_id'].startswith('automation:')
                and json.loads(row['request_id'])[0] == 'kanban-wake')
    except (ValueError, TypeError, IndexError, KeyError):
        return False

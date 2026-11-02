"""``bot_delivery.list``: the teammate deliveries a bot is waiting on, for its Bot Chat strip.

Read-only over the target profiles' delivery receipts (``tools/bot_delivery_status.py``), joined
with each target authority this process serves: the admission's durable state (queued vs. its turn
running) and any prompt the target's turn has parked on the user. The connection's own profile is
the sender, so a bot only ever lists its own requests. Nothing is written or resent.
"""
import asyncio
from functools import partial
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError


def handlers(connection):
    return {'bot_delivery.list': partial(bot_delivery_list, connection)}


def _target_live_state(runner, _name, home, record):
    """``(admission_status, waiting_on)`` read through the target's own authority under ITS
    profile scope; the durable store alone when this process does not serve that profile."""
    from gateway.session_authorities import authority_for_home, owner_scope
    from hermes_state_runtime import get_session_admission
    from tools.bot_delivery_status import stored_live_state

    target = authority_for_home(runner, home) if runner is not None else None
    if target is None or getattr(target, 'retired', False):
        return stored_live_state(_name, home, record)
    admission_id = record.get('retry_admission_id') or record.get('admission_id') or ''
    with owner_scope(target):
        row = get_session_admission(target.db, admission_id=admission_id) if admission_id else None
        live = target.sessions.get(record.get('session_id') or '')
        kinds = live.controls.waiting_kinds() if live is not None and getattr(live, 'controls', None) else []
    # One open prompt is enough to say "waiting on you"; a clarify is named before an approval.
    return (row['status'] if row else None), (kinds[0] if kinds else None)


async def bot_delivery_list(connection, _ref, params):
    from gateway.session_authorities import require_serving, served_profile_name
    from tools.bot_delivery_status import list_sender_deliveries
    from tools.bot_mode_probe import _hermes_root

    authority, actor = connection.authority, connection.actor
    require_serving(authority)
    if 'session:read' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    if set(params) - {'profile'}:
        raise RuntimeStoreError('invalid_params')
    home = await asyncio.to_thread(lambda: Path(authority.db.db_path).parent.resolve())
    sender = served_profile_name(home)
    live_state = partial(_target_live_state, getattr(authority, 'runner', None))
    rows = await asyncio.to_thread(list_sender_deliveries, _hermes_root(home), sender, live_state=live_state)
    return {'sender': sender, 'deliveries': rows}

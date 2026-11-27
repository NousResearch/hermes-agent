"""Generation-fenced controls for a plugin's own admission, never its background children."""
from gateway.session_busy_controls import adopts_owner_agent
from gateway.session_contract import SessionActor
from persistence.core.errors import AuthorityRefusal


def observation(authority, receipt):
    live = authority.sessions.get(receipt.ref.session_id)
    generation = receipt.execution_generation
    active = (receipt.status == 'started' and live is not None
              and authority.executing.get(receipt.ref.session_id) == generation)
    controls = ['cancel'] if receipt.status == 'queued' else []
    if active:
        controls = ['interrupt']
        if adopts_owner_agent(authority, receipt.ref):
            controls.insert(0, 'steer')
    return dict(authority_epoch=authority.epoch, execution_generation=generation,
                progress=(live.event_stream.progress_for(generation) or {}) if active else {}, controls=controls)


def validate(params):
    action = params.get('action')
    if (action not in ('cancel', 'steer', 'interrupt')
            or type(params.get('authority_epoch')) is not int
            or (action != 'cancel' and type(params.get('execution_generation')) is not int)
            or (action == 'steer' and (not isinstance(params.get('text'), str) or not params['text'].strip()))
            or (action != 'steer' and params.get('text') is not None)):
        raise AuthorityRefusal('invalid_params')
    return action


async def apply(authority, request, record):
    from gateway.session_authority import admission_row

    params = request.control
    action = validate(params)
    receipt = authority._receipt(admission_row(record))
    ref = receipt.ref
    if params['authority_epoch'] != authority.epoch:
        raise AuthorityRefusal('stale_generation')
    facts = observation(authority, receipt)
    if (action not in facts['controls']
            or params.get('execution_generation') != receipt.execution_generation):
        raise AuthorityRefusal('stale_generation')
    # The caller has already established profile, current chat and exact plugin/request ownership.
    actor = SessionActor('automation:' + request.session_key, authority.profile_id,
                         frozenset({'session:control', 'session:submit'}), 'plugin:' + request.plugin_id)
    if action == 'cancel':
        await authority.cancel_queued(actor, ref, receipt.admission_id)
        return dict(action=action, status='accepted', delivery='none')
    authority.authorize(actor, ref, 'session:control')
    if action == 'interrupt':
        await authority.interrupt(actor, ref, receipt.execution_generation)
        delivered = authority.delivered_stops.get(ref.session_id) == receipt.execution_generation
        pending = authority.pending_stops.get(ref.session_id) == receipt.execution_generation
        return dict(action=action, status='accepted' if delivered or pending else 'rejected',
                    delivery='agent' if delivered else 'pending' if pending else 'none')
    live = authority.sessions[ref.session_id]
    with live.event_stream.lock:
        authority.check_approval_generation(ref.session_id, receipt.execution_generation)
        agent, latched = authority.steer_agent(ref, receipt.execution_generation, params['text'])
        if latched:
            return dict(action=action, status='accepted', delivery='pending')
        if agent is None:
            return dict(action=action, status='rejected', delivery='none')
        accepted = agent.steer(params['text'])
        return dict(action=action, status='accepted' if accepted else 'rejected', delivery='agent')

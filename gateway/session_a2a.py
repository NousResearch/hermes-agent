"""Profile forwarding through the canonical owner, with exact conversation identity."""
import asyncio
from contextvars import ContextVar
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from gateway.session_contract import SessionRef, Submission
from hermes_state_runtime import RuntimeStoreError

_forward_identity = ContextVar('a2a_forward_identity', default=None)


def build_forward_policy(params, config, *, private_secrets):
    """Only the forwarding handler can mint this policy, never session.create JSON."""
    from gateway.session_policy import build_policy
    identity = _forward_identity.get()
    if identity is None or set(params) != {'source', 'request_id'}:
        raise RuntimeStoreError('invalid_params')
    policy = build_policy(dict(params, source='cli'), config, private_secrets=private_secrets)
    request = json.loads(policy.request_json)
    request.update(source='a2a', a2a_identity=identity)
    return replace(policy, source='a2a', platform='a2a', request_json=json.dumps(request, sort_keys=True))


def is_forward_policy(policy):
    if policy.source != 'a2a' or policy.platform != 'a2a':
        return False
    try:
        request = json.loads(policy.request_json)
        identity = request['a2a_identity']
        return (request['source'] == 'a2a' and isinstance(identity, list) and len(identity) == 4
                and all(isinstance(value, str) for value in identity))
    except (ValueError, KeyError, TypeError):
        return False


def storage_source(db, source, session_id, fallback):
    """Retain the owner's native label when its transport refreshes peer metadata."""
    from gateway.config import Platform
    if source.platform != Platform.LOCAL or not str(source.chat_id).startswith('local-'):
        return fallback
    from hermes_state_local import local_receipt
    receipt = local_receipt(db, source.chat_id)
    if session_id not in receipt.get('lineage', [receipt['session_id']]):
        raise RuntimeStoreError('admission_conflict')
    return 'a2a' if receipt['policy']['source'] == 'a2a' else fallback


async def forward(connection, params):
    authority, actor = connection.authority, connection.actor
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if not {'session:create', 'session:submit', 'session:read'} <= actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    fields = {'agent', 'tenant', 'peer', 'context_id', 'input_id', 'text'}
    if (set(params) != fields or any(not isinstance(params[k], str) for k in fields)
            or any(not params[k] for k in fields - {'tenant'})
            or any(len(params[k]) > 1024 for k in fields - {'text'})
            or len(params['text']) > 1_048_576):
        raise RuntimeStoreError('invalid_params')
    identity = [params[k] for k in ('agent', 'tenant', 'peer', 'context_id')]
    key = 'a2a:' + hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()
    from gateway.session_local_recovery import local_identity, restore_local_session
    from gateway.session_local import create_local_session
    from hermes_state_local import local_receipt
    sid = local_identity(authority.profile_id, actor.subject, key)
    authority._require_admission_open()
    if authority.db.get_session(sid) is not None:
        receipt = local_receipt(authority.db, sid)
        if json.loads(receipt['policy']['request_json']).get('a2a_identity') != identity:
            raise RuntimeStoreError('admission_conflict')
        ref = restore_local_session(authority, sid)
    else:
        # Historical titles discarded peer identity and collapsed context characters.
        # Never silently adopt somebody else's history or create a replacement for it.
        from plugins.platforms.a2a.adapter import _safe_context_slug
        legacy = authority.db.get_session_by_title(f'a2a-{params["agent"]}-{_safe_context_slug(params["context_id"])}')
        if legacy is not None:
            raise RuntimeStoreError('runtime_coordination_required')
        token = _forward_identity.set(identity)
        try:
            ref = create_local_session(authority, actor, {'source': 'a2a', 'request_id': key})
        finally:
            _forward_identity.reset(token)
    if ref.session_id not in connection.subscriptions:
        await connection.resume(ref, {})
    receipt = await authority.submit(actor, Submission(params['input_id'], ref, {'text': params['text']}, 'queue'))
    from gateway.session_results import admission_result
    saved = admission_result(authority.db, receipt.admission_id)
    return dict(asdict(receipt), session_id=ref.session_id,
                result=saved['result'] if saved is not None else None)


async def forward_to_owner(home, *, agent, tenant, peer, context_id, input_id, text, timeout):
    """Transport-only caller; a timeout never cancels or re-identifies accepted work."""
    from hermes_cli.gateway_client import GatewayClient, GatewayClientError, _session_ticket
    from hermes_cli.gateway_runtime import ensure_gateway_runtime
    from websockets.asyncio.client import connect
    home = Path(home).resolve()
    async with asyncio.timeout(timeout):
        ready = await asyncio.to_thread(ensure_gateway_runtime, home)
        if ready.state != 'ready' or ready.endpoint is None:
            raise GatewayClientError(f'gateway_{ready.state}:{ready.reason_code or "not_ready"}')
        ticket = await asyncio.to_thread(_session_ticket, home, ready.endpoint)
        url = ready.endpoint.api_origin.replace('https:', 'wss:').replace('http:', 'ws:') + '/api/ws'
        protocols = ['hermes-gateway-v1', 'hermes-gateway-ticket.' + ticket]
        async with connect(url, subprotocols=protocols, open_timeout=10, max_size=8 * 1024 * 1024) as ws:
            if ws.subprotocol != protocols[0]:
                raise GatewayClientError('gateway_protocol_mismatch')
            async with GatewayClient(ws) as client:
                params = dict(agent=agent, tenant=tenant, peer=peer, context_id=context_id, input_id=input_id, text=text)
                while True:
                    receipt = await client.rpc('a2a.forward', **params)
                    if receipt['status'] == 'unknown':
                        raise GatewayClientError('unknown_execution')
                    if receipt['status'] == 'terminal':
                        return receipt
                    # Consume projections while polling the exact durable input receipt.
                    while not client.events.empty():
                        event = client.events.get_nowait()
                        if isinstance(event, Exception):
                            raise event
                    await asyncio.sleep(.1)

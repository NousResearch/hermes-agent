"""ACP guidance projects the canonical owner's receipts, never viewer-local acceptance."""
from acp.exceptions import RequestError
from hermes_cli.gateway_client import GatewayClientError

from agent.active_turn_guidance import CAPABILITY, VERSION
from gateway.session_guidance import PROTOCOL


def prompt_identity(kwargs):
    if CAPABILITY not in kwargs:
        return ''
    metadata = kwargs[CAPABILITY]
    if not isinstance(metadata, dict):
        raise RequestError.invalid_params('active-turn-guidance metadata must be an object')
    value = metadata.get('promptId')
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise RequestError.invalid_params('active-turn-guidance promptId must be a nonempty string of at most 128 characters')
    return value


class GatewayGuidance:
    def __init__(self, agent):
        self.agent = agent
        self.negotiated = False
        self.requested = False
        self.prompts = {}
        self.started = set()

    def observe_owner(self, descriptor):
        self.negotiated = self.requested and PROTOCOL in descriptor.get('capabilities', [])

    async def capability(self, capabilities):
        metadata = getattr(capabilities, 'field_meta', None) or {}
        value = metadata.get(CAPABILITY)
        self.requested = isinstance(value, dict) and type(value.get('version')) is int and value['version'] == VERSION
        if not self.requested:
            return None
        client = await self.agent._client()
        descriptor = await client.rpc('runtime.describe')
        self.observe_owner(descriptor)
        return {CAPABILITY: {'version': VERSION, 'statusLookup': True, 'literalText': True,
                             'terminalReceipts': True}} if self.negotiated else None

    def track(self, session_id, admission_id, prompt_id):
        if self.negotiated and admission_id and prompt_id:
            self.prompts[(session_id, admission_id)] = prompt_id

    async def started_turn(self, session_id, admission_id):
        key = session_id, admission_id
        if key not in self.prompts or key in self.started:
            return
        snapshot = await self.agent._gateway.rpc('session.guidance.turn', session_id=session_id,
                                                admission_id=admission_id)
        if snapshot.get('state') != 'started':
            return
        self.started.add(key)
        if self.agent._conn:
            await self.agent._conn.ext_notification('ai.nous/guidanceTurn', {
                **{k: snapshot[k] for k in ('runtimeInstanceId', 'sessionId', 'turnId')},
                'promptId': self.prompts[key], 'state': 'started'})

    async def terminal(self, session_id, admission_id):
        key = session_id, admission_id
        prompt_id = self.prompts.pop(key, None)
        self.started.discard(key)
        if prompt_id is None:
            return None
        try:
            client = await self.agent._client()
            snapshot = await client.rpc('session.guidance.turn', session_id=session_id, admission_id=admission_id)
        except (GatewayClientError, OSError, TimeoutError):
            # The committed primary result survives an unavailable auxiliary receipt.
            # Absence of a receipt leaves delivery unknown; it never justifies a resend.
            return None
        return {CAPABILITY: {**snapshot, 'promptId': prompt_id}} if snapshot.get('state') == 'ended' else None

    async def request(self, method, params):
        methods = {'ai.nous/steer': 'session.guidance.admit', 'ai.nous/guidanceStatus': 'session.guidance.status'}
        if method not in methods:
            raise RequestError.method_not_found('_' + method)
        if not self.negotiated:
            raise RequestError.method_not_found('_' + method)
        fields = {'sessionId': 'session_id', 'runtimeInstanceId': 'runtime_instance_id',
                  'expectedTurnId': 'expected_turn_id', 'messageId': 'message_id'}
        if method == 'ai.nous/steer':
            fields['text'] = 'text'
        if any(not isinstance(params.get(k), str) for k in fields):
            raise RequestError.invalid_params('guidance identities and text must be strings')
        session_id = params['sessionId']
        if session_id not in self.agent._snapshots:
            raise RequestError.invalid_params('session is not held by this ACP viewer')
        client = await self.agent._client()
        return await client.rpc(methods[method], **{v: params[k] for k, v in fields.items()})

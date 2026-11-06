"""Active-only guidance owned by the canonical execution authority, never its viewers."""
from functools import partial

from agent.active_turn_guidance import GuidanceInbox
from hermes_state_runtime import RuntimeStoreError

PROTOCOL = 'active-turn-guidance-v1'
METHODS = ('session.guidance.turn', 'session.guidance.admit', 'session.guidance.status')


class SessionGuidance:
    """The stream lock protects this ledger; each engine gets its own immutable inbox."""
    def __init__(self, instance_id, session_id):
        self.instance_id, self.session_id = instance_id, session_id
        self.current = None
        self.generation = None
        self.by_message = {}
        self.by_admission = {}

    def start(self, row):
        self.current = GuidanceInbox(self.instance_id, self.session_id)
        self.current.start(row['admission_id'])
        self.generation = row['generation']

    def request(self, params, *, lookup):
        identity = {'runtimeInstanceId': params['runtime_instance_id'], 'sessionId': self.session_id,
                    'expectedTurnId': params['expected_turn_id'], 'messageId': params['message_id']}
        if not lookup:
            identity['text'] = params['text']
        inbox = self.by_message.get(params['message_id'], self.current)
        if inbox is None:
            return rejection(identity, 'unknown_message' if lookup else 'rejected_turn_ended')
        if not lookup and params['message_id'] not in self.by_message and len(self.by_message) >= 4096:
            return rejection(identity, 'receipt_capacity_reached')
        result = inbox.request(identity, lookup=lookup)
        if result['disposition'] == 'accepted_active':
            self.by_message[params['message_id']] = inbox
            self.by_admission[inbox.prompt_id] = inbox
        return result

    def snapshot(self, admission_id):
        inbox = self.by_admission.get(admission_id)
        if self.current is not None and self.current.prompt_id == admission_id:
            inbox = self.current
        return inbox.snapshot() if inbox is not None else {'state': 'unavailable'}


def rejection(identity, disposition):
    return {**{k: v for k, v in identity.items() if k not in {'text', 'expectedTurnId'}},
            'turnId': identity['expectedTurnId'], 'disposition': disposition, 'state': 'rejected'}


def start(authority, ref, row):
    """Called while stamping the committed primary claim under the stream lock."""
    from gateway.session_policy import policy_for_source
    from gateway.session_busy_controls import adopts_owner_agent
    live = authority.sessions[ref.session_id]
    policy = policy_for_source(authority.runner, live.source)
    if policy is None or policy.source != 'acp' or not adopts_owner_agent(authority, ref):
        return
    if live.guidance is None:
        live.guidance = SessionGuidance(f'{authority.instance_id}:{authority.epoch}', ref.session_id)
    live.guidance.start(row)


def close(live, generation, *, cancelled=False):
    state = live.guidance
    if state is not None and state.generation == generation and state.current is not None:
        state.current.close(cancelled=cancelled)


def adopt(live, generation, agent):
    state = live.guidance if live is not None else None
    agent._active_turn_guidance = state.current if state is not None and state.generation == generation else None


def handlers(connection):
    return {METHODS[0]: partial(turn, connection), METHODS[1]: partial(request, connection, lookup=False),
            METHODS[2]: partial(request, connection, lookup=True)}


def live_session(connection, ref, capability):
    # Status after retirement/restart must never construct another execution owner.
    if ref.session_id not in connection.authority.sessions:
        raise RuntimeStoreError('not_found')
    connection.authority.authorize(connection.actor, ref, capability)
    return connection.authority.sessions[ref.session_id]


async def turn(connection, ref, params):
    live = live_session(connection, ref, 'session:read')
    with live.event_stream.lock:
        return live.guidance.snapshot(params['admission_id']) if live.guidance else {'state': 'unavailable'}


async def request(connection, ref, params, *, lookup):
    live = live_session(connection, ref, 'session:read' if lookup else 'session:control')
    with live.event_stream.lock:
        if live.guidance is None:
            identity = {'runtimeInstanceId': params['runtime_instance_id'], 'sessionId': ref.session_id,
                        'expectedTurnId': params['expected_turn_id'], 'messageId': params['message_id']}
            return rejection(identity, 'unknown_message' if lookup else 'rejected_turn_ended')
        return live.guidance.request(params, lookup=lookup)

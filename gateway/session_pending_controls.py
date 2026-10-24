"""Generation-bound projection/control of the existing blocking approval queue.

Ordinary approvals and clarify prompts share the existing native waiters.
Secret/sudo prompts remain native; no response is published or journaled here.
"""
from copy import deepcopy

from hermes_state_runtime import RuntimeStoreError
from tools.approval import list_gateway_approvals, resolve_gateway_approval


class PendingControls:
    def __init__(self, events):
        self.events = events
        self.pending = {}
        self.remote_responders = {}

    def register(self, session_id, route, generation, data, *, timeout_s=None):
        from gateway.run import _redact_approval_command
        with self.events.lock:
            prompt_id = data['request_id']
            choices = ['once', 'deny']
            if data.get('allow_session', True) and not data.get('smart_denied', False):
                choices.append('session')
            if data.get('allow_permanent', True) and not data.get('smart_denied', False):
                choices.append('always')
            prompt = {'kind': 'approval', 'prompt_id': prompt_id,
                      'execution_generation': generation,
                      'command': _redact_approval_command(data.get('command', '')),
                      'description': _redact_approval_command(data.get('description', '')),
                      'choices': choices}
            if 'edit' in data:
                prompt['edit'] = deepcopy(data['edit'])
                prompt['choices'] = ['once', 'deny']
            if timeout_s is not None:
                prompt['timeout_s'] = timeout_s
            self.pending[prompt_id] = (route, prompt)
            self.events.publish(session_id, prompt, event_type='approval.request')

    def register_clarify(self, session_id, generation, entry, *, timeout_s=None):
        from gateway.run import _redact_approval_command
        with self.events.lock:
            prompt = {"kind": "clarify", "prompt_id": entry.clarify_id,
                      "execution_generation": generation,
                      "question": _redact_approval_command(entry.question),
                      "choices": [_redact_approval_command(c) for c in entry.choices or []],
                      "multi_select": entry.multi_select}
            if timeout_s is not None:
                prompt["timeout_s"] = timeout_s
            self.pending[entry.clarify_id] = (entry, prompt)
            self.events.publish(session_id, prompt, event_type="clarify.request")

    def snapshot(self, session_id, generation):
        with self.events.lock:
            for prompt_id, (route, prompt) in list(self.pending.items()):
                active = (prompt_id in self.remote_responders or
                          (not route.event.is_set() if prompt['kind'] == 'clarify' else
                           any(p['request_id'] == prompt_id for p in list_gateway_approvals(route))))
                if prompt['execution_generation'] != generation or not active:
                    del self.pending[prompt_id]
                    self.remote_responders.pop(prompt_id, None)
                    self.events.publish(session_id, {'prompt_id': prompt_id,
                        'execution_generation': prompt['execution_generation']}, event_type=prompt['kind'] + '.settled')
            return tuple(deepcopy(prompt) for _, prompt in self.pending.values())

    def presentation(self, generation, prompt_id, route_key, viewer, state):
        from tools.clarify_gateway import record_presentation
        with self.events.lock:
            pending = self.pending.get(prompt_id)
            if pending is None:
                return 'already_resolved'
            entry, prompt = pending
            if (prompt['kind'] != 'clarify' or prompt['execution_generation'] != generation
                    or prompt_id in self.remote_responders or entry.session_key != route_key):
                return 'ineligible'
            return record_presentation(entry, viewer, state)

    def withdraw_presenter(self, viewer):
        from tools.clarify_gateway import record_presentation
        with self.events.lock:
            for prompt_id, (entry, prompt) in self.pending.items():
                if prompt['kind'] == 'clarify' and prompt_id not in self.remote_responders:
                    record_presentation(entry, viewer, 'hidden')

    def respond(self, session_id, generation, prompt_id, response, *, kind="approval"):
        with self.events.lock:
            self.snapshot(session_id, generation)
            entry = self.pending.get(prompt_id)
            if entry is None:
                return {'status': 'already_resolved', 'prompt_id': prompt_id}
            route, prompt = entry
            if prompt['kind'] != kind:
                raise RuntimeStoreError('invalid_params')
            if prompt_id in self.remote_responders:
                field = 'answer' if kind == 'clarify' else 'choice'
                if (not isinstance(response, dict) or set(response) != {field}
                        or not isinstance(response[field], str) or len(response[field]) > 16384
                        or (kind == 'approval' and response[field] not in prompt['choices'])):
                    raise RuntimeStoreError('invalid_params')
                self.remote_responders[prompt_id](kind, prompt_id, response[field])
                del self.remote_responders[prompt_id]
                del self.pending[prompt_id]
                self.events.publish(session_id, {'prompt_id': prompt_id, 'execution_generation': generation},
                                    event_type=kind + '.settled')
                return {'status': 'resolved', 'prompt_id': prompt_id}
            if kind == 'clarify':
                from tools.clarify_gateway import resolve_gateway_clarify
                if (not isinstance(response, dict) or set(response) != {'answer'}
                        or not isinstance(response['answer'], str)):
                    raise RuntimeStoreError('invalid_params')
                resolved = resolve_gateway_clarify(prompt_id, response['answer'])
                self.snapshot(session_id, generation)
                return {'status': 'resolved' if resolved else 'already_resolved', 'prompt_id': prompt_id}
            if not isinstance(response, dict) or set(response) != {'choice'} or response['choice'] not in prompt['choices']:
                raise RuntimeStoreError('invalid_params')
            # The tools lock arbitrates native taps versus attached responders;
            # an exact request ID can never consume the next FIFO approval.
            resolved = resolve_gateway_approval(route, response['choice'], request_id=prompt_id)
            self.snapshot(session_id, generation)
            return {'status': 'resolved' if resolved else 'already_resolved', 'prompt_id': prompt_id}


async def clarify_presentation(connection, ref, params):
    """Accept UI evidence only from the exact current local viewer and prompt."""
    from gateway.config import Platform
    authority, actor = connection.authority, connection.actor
    authority.authorize(actor, ref, 'session:respond')
    live = authority.sessions[ref.session_id]
    with live.event_stream.lock:
        if (actor not in live.subscribers.values()
                or authority.events.get(actor.transport_id) is not connection.transport):
            raise RuntimeStoreError('permission_denied')
        generation = params['execution_generation']
        if type(generation) is not int:
            raise RuntimeStoreError('stale_generation')
        authority.check_approval_generation(ref.session_id, generation)
        if params['state'] not in {'presented', 'interaction', 'hidden'}:
            raise RuntimeStoreError('invalid_params')
        status = 'ineligible'
        if live.source is not None and live.source.platform == Platform.LOCAL:
            status = live.controls.presentation(generation, params['prompt_id'], live.route,
                                                actor.transport_id, params['state'])
        return {'status': status, 'prompt_id': params['prompt_id']}

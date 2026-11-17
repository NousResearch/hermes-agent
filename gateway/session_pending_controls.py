"""Generation-bound projection/control of the existing blocking approval queue.

Ordinary approvals and clarify prompts share the existing native waiters.
Secret/sudo prompts remain native; no response is published or journaled here.
"""
from copy import deepcopy
from functools import partial

from gateway.session_asks import INTERRUPTED, Ask, AskLedger, approval_outcome, prompt_payload, question_outcome
from hermes_state_runtime import RuntimeStoreError
from tools.approval import list_gateway_approvals, observe_gateway_outcome, resolve_gateway_approval
from tools.clarify_gateway import SKIPPED as CLARIFY_SKIPPED


class PendingControls:
    def __init__(self, events, asks=None):
        self.events = events
        # Where the prompts are recorded as asks: the live session shares its event stream's ledger,
        # so its secret requests and these prompts list together; a standalone one keeps its own.
        self.asks = asks if asks is not None else AskLedger()
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
            self._open_ask(session_id, 'approval', prompt, route)
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
            self._open_ask(session_id, 'question', prompt, entry)
            self.events.publish(session_id, prompt, event_type="clarify.request")

    def _open_ask(self, session_id, kind, prompt, route):
        """Record the prompt as an ask (gateway/session_asks.py) and how its native waiter reports its end."""
        prompt_id = prompt['prompt_id']
        ask = Ask(prompt_id, kind, session_id, prompt['execution_generation'], prompt_payload(prompt))
        if prompt_id in self.remote_responders:
            # A managed worker's prompt: its waiter lives in the worker. ``respond`` settles an answer;
            # a prompt the worker withdrew leaves ``remote_responders`` unanswered.
            ask.probe = lambda: None if prompt_id in self.remote_responders else INTERRUPTED
        elif kind == 'question':
            ask.probe = lambda: (question_outcome(getattr(route, 'response', None))
                                 if route.event.is_set() else None)
        self.asks.open(ask)
        if kind == 'approval' and ask.probe is None:
            settle = partial(self.asks.settle, prompt_id)
            if not observe_gateway_outcome(route, prompt_id,
                                           lambda reason, outcome: settle(*approval_outcome(reason, outcome))):
                settle(*INTERRUPTED)

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
                value = response[field]
                self.asks.settle(prompt_id, *(approval_outcome('resolved', {'choice': value}) if kind == 'approval'
                                              else question_outcome(value if value.strip() else CLARIFY_SKIPPED)))
                del self.remote_responders[prompt_id]
                del self.pending[prompt_id]
                self.events.publish(session_id, {'prompt_id': prompt_id, 'execution_generation': generation},
                                    event_type=kind + '.settled')
                return {'status': 'resolved', 'prompt_id': prompt_id}
            if kind == 'clarify':
                from tools.clarify_gateway import SKIPPED, resolve_gateway_clarify
                if (not isinstance(response, dict) or set(response) != {'answer'}
                        or not isinstance(response['answer'], str)):
                    raise RuntimeStoreError('invalid_params')
                # An empty answer is the documented skip; the waiter reads a bare "" as a timeout.
                resolved = resolve_gateway_clarify(prompt_id, response['answer'] if response['answer'].strip() else SKIPPED)
                self.snapshot(session_id, generation)
                return {'status': 'resolved' if resolved else 'already_resolved', 'prompt_id': prompt_id}
            if not isinstance(response, dict) or set(response) != {'choice'} or response['choice'] not in prompt['choices']:
                raise RuntimeStoreError('invalid_params')
            # The tools lock arbitrates native taps versus attached responders;
            # an exact request ID can never consume the next FIFO approval.
            resolved = resolve_gateway_approval(route, response['choice'], request_id=prompt_id)
            self.snapshot(session_id, generation)
            return {'status': 'resolved' if resolved else 'already_resolved', 'prompt_id': prompt_id}

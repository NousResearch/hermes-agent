"""Prepared native controls retain their identity across ambiguous RPC replies."""
import json
import uuid
from hermes_cli.gateway_client import GatewayClientError


class PreparedMutations:
    def __init__(self):
        self.pending = {}

    async def apply(self, client, session_id, operation, payload):
        encoded = json.dumps(payload, sort_keys=True)
        key = (session_id, operation, encoded)
        if key not in self.pending:
            snapshot = await client.rpc('session.resume', session_id=session_id)
            self.pending[key] = {'params': dict(session_id=session_id, operation=operation,
                payload=json.loads(encoded), request_id=uuid.uuid4().hex,
                expected_revision=snapshot['revision'], expected_generation=snapshot['execution_generation'])}
        entry = self.pending[key]
        if 'result' not in entry:
            try:
                entry['result'] = await client.rpc('session.mutate', **entry['params'])
            except GatewayClientError as exc:
                # A disconnect is not a definitive authority refusal. Preserve
                # the tuple; never refresh its preconditions behind the user.
                if str(exc) in {'invalid_params', 'revision_conflict', 'stale_generation',
                                'session_busy', 'unknown_execution', 'permission_denied',
                                'model_resolution_failed', 'nothing_to_compress'}:
                    self.pending.pop(key)
                raise
        return entry['result']

    def acknowledge(self, session_id, operation, payload):
        self.pending.pop((session_id, operation, json.dumps(payload, sort_keys=True)), None)


def slash_mutation(command, arg):
    name = command.lstrip('/')
    if name == 'model':
        from hermes_cli.model_switch import parse_model_flags_detailed
        parsed = parse_model_flags_detailed(arg)
        if parsed.is_global or parsed.is_once or parsed.force_refresh or not parsed.model_input:
            raise GatewayClientError('unsupported_model_options')
        return name, {'model': parsed.model_input, **(
            {'provider': parsed.explicit_provider} if parsed.explicit_provider else {})}
    fields = {'branch': 'title', 'compress': 'focus'}
    if name not in fields:
        raise GatewayClientError('unsupported_command')
    return name, {fields[name]: arg} if arg else {}

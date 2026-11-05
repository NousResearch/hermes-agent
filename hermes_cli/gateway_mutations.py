"""Prepared native controls retain their identity across ambiguous RPC replies."""
import json
import uuid
from hermes_cli.gateway_client import COMPRESS_RPC_TIMEOUT, GatewayClientError


class PreparedMutations:
    def __init__(self):
        self.pending = {}

    async def apply(self, client, session_id, operation, payload, *, confirm=None):
        """The owner's answer for this prepared mutation (retained across ambiguous replies).

        A guarded model target answers ``status: confirmation_required`` and writes nothing.
        ``confirm`` (``async (refusal) -> bool``) is the surface's prompt: yes re-sends the same
        retained mutation once with ``payload.confirm`` = the owner's token (under the original
        key, so an ambiguous reply to the confirmed send retries that exact request); no answers
        ``status: cancelled``. Without a prompt (non-interactive) the refusal is an error, and a
        confirmed send refused again (another switch or a turn landed first) is never re-asked.

        ``payload`` may be a function of the snapshot the CAS tuple is read from (a rewind names
        its target row from that same transcript); returning None means "nothing to do" and is
        answered ``status: nothing``. A derived payload is keyed by operation alone, so a lost-reply
        retry re-presents the original target instead of re-deriving one from an already-rewound
        transcript. Metadata edits (``rename``) carry no generation fence, as on Ink/Desktop."""
        key = _key(session_id, operation, payload)
        self.retire(session_id, keep=key)
        if key not in self.pending:
            snapshot = await client.rpc('session.resume', session_id=session_id)
            body = payload(snapshot) if callable(payload) else json.loads(key[2])
            if body is None:
                return {'status': 'nothing', 'session_id': session_id, 'operation': operation}
            fence = {} if operation in _METADATA else {'expected_generation': snapshot['execution_generation']}
            self.pending[key] = {'params': dict(session_id=session_id, operation=operation, payload=body,
                request_id=uuid.uuid4().hex, expected_revision=snapshot['revision'], **fence)}
        entry = self.pending[key]
        while 'result' not in entry:
            try:
                # A compression answers after its summary + commit; the default budget would report
                # a timeout while the owner (which shields the mutation) still commits it.
                budget = {'_timeout': COMPRESS_RPC_TIMEOUT} if operation == 'compress' else {}
                entry['result'] = await client.rpc('session.mutate', **budget, **entry['params'])
            except GatewayClientError as exc:
                # A disconnect is not a definitive authority refusal. Preserve
                # the tuple; never refresh its preconditions behind the user.
                if str(exc) in {'invalid_params', 'revision_conflict', 'stale_generation',
                                'session_busy', 'unknown_execution', 'permission_denied',
                                'model_resolution_failed', 'nothing_to_compress',
                                'unsupported_compress_options'}:
                    self.pending.pop(key)
                raise
            refusal = entry['result']
            if refusal.get('status') != 'confirmation_required':
                break
            # The owner wrote nothing (no receipt): the same tuple may be re-sent with the token.
            del entry['result']
            if confirm is None or 'confirm' in entry['params']['payload']:
                self.pending.pop(key)
                raise GatewayClientError(model_refusal_text(refusal, confirmed=confirm is not None))
            try:
                accepted = await confirm(refusal)
            except BaseException:
                self.pending.pop(key)
                raise
            if not accepted:
                self.pending.pop(key)
                return {'status': 'cancelled', 'session_id': session_id, 'operation': operation}
            entry['params'] = {**entry['params'], 'request_id': uuid.uuid4().hex,
                               'payload': {**entry['params']['payload'], 'confirm': refusal['confirm']}}
        return entry['result']

    def acknowledge(self, session_id, operation, payload):
        self.pending.pop(_key(session_id, operation, payload), None)

    def retire(self, session_id, *, keep=None):
        """Only a session's very next control may be the retry of an ambiguous one: any other verb
        on the session (a different control, a prompt) retires its retained identities, so a later
        deliberate repeat is new work instead of replaying a stale receipt."""
        for key in [key for key in self.pending if key[0] == session_id and key != keep]:
            del self.pending[key]


_METADATA = frozenset({'rename'})


def _key(session_id, operation, payload):
    return (session_id, operation, None if callable(payload) else json.dumps(payload, sort_keys=True))


def model_refusal_text(refusal, *, confirmed=False):
    """Why a guarded model target was not applied (the owner's guard text first)."""
    reason = ('the session changed before the confirmation landed; run /model again.' if confirmed else
              'this target needs a confirmation; run /model in an interactive session to confirm it.')
    return f"{refusal['confirm_message']}\n\nModel not switched: {reason}"


def confirmation_title(refusal):
    """The one prompt title for the owner's guard warnings (as ``combined_selection_warning``)."""
    warnings = refusal.get('warnings') or []
    return warnings[0]['title'] if len(warnings) == 1 else 'Model Selection Warning'


def confirm_choice(raw, choices):
    """``once`` / ``cancel`` (or None) for a typed answer: the choice's number or the classic
    CLI's confirm aliases (y/yes/once, n/no/cancel)."""
    from hermes_cli.cli_modal_mixin import _CONFIRM_ALIASES
    answer = (raw or '').strip().lower()
    if answer.isdigit() and 0 < int(answer) <= len(choices):
        return choices[int(answer) - 1][0]
    allowed = {choice[0] for choice in choices}
    normalized = _CONFIRM_ALIASES.get(answer, answer)
    return normalized if normalized in allowed else None


def compress_payload(arg):
    """Structured ``session.mutate(compress)`` payload from the raw ``/compress`` arguments.

    The shared parser is the one every native surface uses, so ``--preview`` stays a read-only
    flag and ``here [N]`` a boundary instead of becoming a focus topic; ``--aggressive`` has no
    canonical implementation and is refused before anything reaches the authority.
    """
    from agent.conversation_compression_manual import AGGRESSIVE_UNSUPPORTED, parse_compress_args
    request = parse_compress_args(arg)
    if request.aggressive:
        raise GatewayClientError(AGGRESSIVE_UNSUPPORTED)
    payload = {}
    if request.focus_topic:
        payload['focus'] = request.focus_topic
    if request.preview:
        payload['preview'] = True
    if request.partial:
        payload.update(partial=True, keep_last=request.keep_last)
    return payload


def slash_mutation(command, arg):
    name = command.lstrip('/')
    if name == 'model':
        from hermes_cli.model_switch import parse_model_flags_detailed
        parsed = parse_model_flags_detailed(arg)
        if parsed.is_global or parsed.is_once or parsed.force_refresh or not parsed.model_input:
            raise GatewayClientError('unsupported_model_options')
        return name, {'model': parsed.model_input, **(
            {'provider': parsed.explicit_provider} if parsed.explicit_provider else {})}
    if name == 'compress':
        return name, compress_payload(arg)
    if name != 'branch':
        raise GatewayClientError('unsupported_command')
    return name, {'title': arg} if arg else {}

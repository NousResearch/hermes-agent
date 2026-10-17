"""Per-admission turn inputs the classic CLI sends with ``prompt.submit`` (CALLMAP W2).

Main's HermesCLI calls ``AIAgent.run_conversation`` with an API-only message (voice prefix,
model-switch / skills-reload notes) while persisting the clean text, names the turn's author for
memory attribution, and runs ``/model --once`` and ``/moa`` as a one-turn runtime swap. Over the
session authority those are facts about ONE admitted turn, committed with the admission row as
``cli_turn_v1`` so a retry or a queued drain runs exactly what was admitted, and bound for that
admission's execution only, never for a later turn.
"""
from contextlib import contextmanager
from contextvars import ContextVar

from hermes_state_runtime import RuntimeStoreError

_SUBMIT_FIELDS = ('turn_author', 'api_text', 'model_override', 'clarify_batch')
_API_TEXT_LIMIT = 1_000_000
_cli_turn: ContextVar[tuple | None] = ContextVar('cli_turn', default=None)


def submit_cli_turn_fields(params):
    """The raw per-turn CLI fields of a ``prompt.submit`` request, forwarded to admission."""
    return {key: params[key] for key in _SUBMIT_FIELDS if key in params}


def admit_cli_turn(params):
    """Wire fields -> committed ``cli_turn_v1`` (``{}`` when absent); malformed values are refused."""
    committed = {}
    if params.get('turn_author') is not None:
        from agent.turn_author import parse_turn_author
        author = parse_turn_author(params['turn_author'])
        if author is None:
            raise RuntimeStoreError('invalid_params')
        committed['turn_author'] = author
    api_text = params.get('api_text')
    if api_text is not None:
        if not isinstance(api_text, str) or len(api_text) > _API_TEXT_LIMIT:
            raise RuntimeStoreError('invalid_params')
        if api_text:
            committed['api_text'] = api_text
    override = params.get('model_override')
    if override is not None:
        if (not isinstance(override, dict) or set(override) - {'model', 'provider'}
                or not isinstance(override.get('model'), str) or not override['model'].strip()
                or not isinstance(override.get('provider', ''), str)):
            raise RuntimeStoreError('invalid_params')
        committed['model_override'] = {'model': override['model'].strip(),
                                       **({'provider': override['provider']} if override.get('provider') else {})}
    batch = params.get('clarify_batch')
    if batch is not None and type(batch) is not bool:
        raise RuntimeStoreError('invalid_params')
    if batch:
        # The submitting viewer is main's HermesCLI, which answers the clarify tool's whole batch
        # in one ``clarify`` server request (its batch panel). Other viewers of a ``cli``-source
        # session (an ACP editor, a raw WebSocket client) answer per-question cards.
        committed['clarify_batch'] = True
    return {'cli_turn_v1': committed} if committed else {}


@contextmanager
def cli_turn_scope(admission_id, committed):
    token = _cli_turn.set((admission_id, committed) if committed else None)
    try:
        yield
    finally:
        _cli_turn.reset(token)


def cli_turn(message_id):
    """The committed CLI turn fields of the admission whose event id is ``message_id``; ``{}`` for
    any other turn, so a follow-up drained on the same task never inherits them."""
    bound = _cli_turn.get()
    if bound is None or message_id is None or str(message_id) != bound[0]:
        return {}
    return bound[1]


def rebind_api_text(message_id, text):
    """This admission's ``api_text`` with its ``@`` references expanded, for the rest of this execution
    (``session_context_refs.turn_references``). The committed row keeps what the CLI sent, so a retry
    expands it afresh rather than twice."""
    bound = _cli_turn.get()
    if bound is not None and message_id is not None and str(message_id) == bound[0]:
        _cli_turn.set((bound[0], {**bound[1], 'api_text': text}))


def one_turn_runtime(override, model, runtime_kwargs, config):
    """``(model, runtime_kwargs)`` of a committed ``model_override`` for this one turn, resolved by the
    same ``switch_model`` pipeline ``/model`` uses against the session's current route. Nothing is
    written: the session's route, policy and cached agent are untouched, so the next turn runs its own
    route again (main's ``/model --once`` restore). Raises ``RuntimeError`` with the switch's own
    sentence when the target cannot be resolved; the caller fails the turn with it."""
    from hermes_cli.config import get_compatible_custom_providers
    from hermes_cli.model_switch import switch_model
    result = switch_model(
        raw_input=override['model'], current_provider=runtime_kwargs.get('provider') or '',
        current_model=model or '', current_base_url=runtime_kwargs.get('base_url') or '',
        current_api_key=runtime_kwargs.get('api_key') or '', is_global=False,
        explicit_provider=override.get('provider', ''), user_providers=(config or {}).get('providers'),
        custom_providers=get_compatible_custom_providers(config or {}))
    if not result.success:
        raise RuntimeError(result.error_message or f"model {override['model']!r} could not be resolved")
    runtime = dict(runtime_kwargs)
    if result.target_provider != runtime_kwargs.get('provider'):
        # The previous provider's pool and ACP command belong to that provider only.
        runtime.update(credential_pool=None, command=None, args=[])
    runtime.update(provider=result.target_provider, requested_provider=result.target_provider,
                   api_key=result.api_key, base_url=result.base_url, api_mode=result.api_mode or None,
                   request_overrides=dict(result.request_overrides or {}),
                   capabilities=dict(result.runtime_capabilities or {}))
    return result.new_model, runtime


# run_conversation keys main's HermesCLI reads that the gateway's own projection does not carry
# (CALLMAP §3.3 step 4): the transform-preview original, the billing descriptor and the turn's
# last reasoning / preview flags.
_RESULT_PASSTHROUGH = ('pre_transform_response', 'billing_block', 'last_reasoning', 'response_previewed',
                       'response_transformed', 'turn_exit_reason')


def classic_cli_turn(turn):
    """A local-viewer turn on a classic-CLI surface session (``cli`` / ``oneshot`` / ``tool``): the
    viewer is main's HermesCLI, which reads the run_conversation-shaped result itself."""
    from gateway.run_turn_progress import local_viewer_turn
    if not local_viewer_turn(turn):
        return False
    from gateway.session_policy import policy_for_source
    try:
        policy = policy_for_source(turn._runner, turn._ctx.source)
    except RuntimeStoreError:  # a route whose frozen policy is not restored: not a CLI viewer turn
        return False
    return policy is not None and policy.platform == 'cli'


def clarify_batch_turn(turn):
    """The clarify tool's batch goes to main's batch panel as one request: a classic-CLI turn that
    the classic CLI itself submitted (``clarify_batch`` committed with its admission)."""
    return classic_cli_turn(turn) and bool(cli_turn(getattr(turn._ctx, 'inbound_message_id', None)).get('clarify_batch'))


def take_cli_result_fields(result, *, aux_before=None, session_db=None, session_id=None):
    """The receipt-only fields of a classic CLI turn, taken from the agent's raw result.

    ``pending_steer`` is REMOVED from ``result``: main's CLI queues a leftover /steer as its own
    next turn (``delivering leftover steer``), so the gateway must not also run it as a follow-up
    inside this admission. ``final_response`` is the agent's raw text: main renders its own
    ``chat_error_copy`` for an empty failed turn instead of the gateway's chat copy."""
    if not isinstance(result, dict):
        return {}
    fields = {key: result.get(key) for key in _RESULT_PASSTHROUGH if key in result}
    fields['final_response'] = result.get('final_response') or ''
    steer = result.pop('pending_steer', None)
    fields['pending_steer'] = steer or None
    if aux_before is not None:
        from hermes_cli.oneshot import _attach_auxiliary_usage
        holder = {'session_id': result.get('session_id') or session_id}
        _attach_auxiliary_usage(holder, session_db, aux_before, fallback_session_id=session_id)
        fields['auxiliary_usage'] = holder.get('auxiliary_usage') or {}
    return fields


def one_turn_route(turn, policy, model, runtime_kwargs):
    """The executing admission's ``model_override`` applied to the resolved route (unchanged without one)."""
    override = cli_turn(turn._ctx.inbound_message_id).get('model_override')
    if not override:
        return model, runtime_kwargs
    from gateway.session_authorities import active_authority
    config = policy.config(active_authority(turn._runner)) if policy else turn._ctx.user_config
    return one_turn_runtime(override, model, runtime_kwargs, config)


def run_classic_cli_turn(turn, run):
    """``run()`` (the turn's run_conversation) for any turn; on a classic CLI turn the receipt-only
    fields are taken from its raw result and kept on ``turn`` for the committed receipt."""
    if not classic_cli_turn(turn):
        return run()
    from gateway.session_finite import finite_turn_required
    aux_before = cli_db = None
    if finite_turn_required() is True:
        # `chat -q --usage-file` / stream-json: this turn's auxiliary spend, as oneshot.py read it.
        from hermes_cli.oneshot import _auxiliary_usage
        runner_db = turn._runner._session_db
        cli_db = getattr(runner_db, "_db", runner_db)
        aux_before = _auxiliary_usage(cli_db, turn._ctx.session_id)
    result = run()
    turn._cli_result_fields = take_cli_result_fields(
        result, aux_before=aux_before, session_db=cli_db, session_id=turn._ctx.session_id)
    return result

"""Admission-scoped finite consumers, independent of viewer or daemon lifetime."""
from contextlib import contextmanager
from contextvars import ContextVar

from hermes_state_runtime import RuntimeStoreError

_finite_turn = ContextVar('finite_turn', default=None)
_unattended_turn = ContextVar('unattended_turn', default=False)


def finite_turn_required():
    # None preserves legacy standalone CLI's marker. A bound owner turn must
    # never inherit process launch flags from another viewer.
    return _finite_turn.get()


FINITE_CLARIFY_NOTICE = (
    "single-query mode: no user available to answer. Pick the best choices using your own "
    "judgment, or make the most reasonable assumption you can, and continue.")


def finite_clarify_reply():
    """clarify_tool's reply on a finite turn (``chat -q`` / ``-z``): main's headless answer."""
    return {"answers": {}, "outcome": "undelivered", "notice": FINITE_CLARIFY_NOTICE}


def unattended_turn():
    # `hermes -z`: nobody can answer a prompt, so the classic one-shot contract auto-approves
    # (legacy oneshot.py set HERMES_YOLO_MODE). `-q` stays single-query (approvals.single_query_mode).
    return _unattended_turn.get()


@contextmanager
def finite_turn_scope(finite, unattended=False):
    token = _finite_turn.set(finite)
    unattended_token = _unattended_turn.set(bool(finite and unattended))
    try:
        yield
    finally:
        _unattended_turn.reset(unattended_token)
        _finite_turn.reset(token)


def admit_finite(params, *, submitter=None):
    """Validate a submission's finite flags. *submitter* ``(connection, ref)`` marks a wire client: it may
    ask for an unattended (auto-approving) turn only as the local CLI principal on its own CLI session."""
    if 'unattended' in params and (type(params['unattended']) is not bool
                                   or (params['unattended'] and params.get('finite') is not True)):
        raise RuntimeStoreError('invalid_params')
    if params.get('unattended') and submitter is not None and not _may_run_unattended(*submitter):
        raise RuntimeStoreError('invalid_params')
    if 'finite' not in params:
        return {}
    if type(params['finite']) is not bool:
        raise RuntimeStoreError('invalid_params')
    return {'finite': params['finite'], **({'unattended': True} if params.get('unattended') else {})}


def _may_run_unattended(connection, ref):
    """``hermes -z`` parity: main auto-approved only inside the one-shot's own process. Over the wire that
    is the private local bootstrap ticket (no dashboard / remote identity), the session's creator (no
    operator attach) and a CLI-surface session (never a Desktop, TUI, ACP or Bot Chat others watch)."""
    from gateway.session_policy import policy_for_source
    source = connection.authority.sessions[ref.session_id].source
    if not connection.native_owner or source is None or source.user_id != connection.actor.subject:
        return False
    policy = policy_for_source(connection.authority.runner, source)
    return policy is not None and policy.platform == 'cli'


async def execute_finite_admission(authority, ref, row):
    from gateway.session_ingress import execute_admission
    from gateway.session_display import display_turn_scope, submission_turn_scope
    from gateway.session_surface import surface_turn_scope
    from gateway.session_cli_turn import cli_turn_scope
    with finite_turn_scope(row['payload'].get('finite', False), row['payload'].get('unattended') is True), \
            surface_turn_scope(row['payload'].get('surface_v1')), \
            display_turn_scope(row['admission_id'], row['payload'].get('display_v1')), \
            submission_turn_scope(row['admission_id'], row.get('request_id')), \
            cli_turn_scope(row['admission_id'], row['payload'].get('cli_turn_v1')):
        return await execute_admission(authority, ref, row)

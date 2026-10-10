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


def admit_finite(params):
    if 'unattended' in params and (type(params['unattended']) is not bool
                                   or (params['unattended'] and params.get('finite') is not True)):
        raise RuntimeStoreError('invalid_params')
    if 'finite' not in params:
        return {}
    if type(params['finite']) is not bool:
        raise RuntimeStoreError('invalid_params')
    return {'finite': params['finite'], **({'unattended': True} if params.get('unattended') else {})}


async def execute_finite_admission(authority, ref, row):
    from gateway.session_ingress import execute_admission
    from gateway.session_display import display_turn_scope
    from gateway.session_surface import surface_turn_scope
    with finite_turn_scope(row['payload'].get('finite', False), row['payload'].get('unattended') is True), \
            surface_turn_scope(row['payload'].get('surface_v1')), \
            display_turn_scope(row['admission_id'], row['payload'].get('display_v1')):
        return await execute_admission(authority, ref, row)

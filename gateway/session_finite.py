"""Admission-scoped finite consumers, independent of viewer or daemon lifetime."""
from contextlib import contextmanager
from contextvars import ContextVar

from hermes_state_runtime import RuntimeStoreError

_finite_turn = ContextVar('finite_turn', default=None)


def finite_turn_required():
    # None preserves legacy standalone CLI's marker. A bound owner turn must
    # never inherit process launch flags from another viewer.
    return _finite_turn.get()


@contextmanager
def finite_turn_scope(finite):
    token = _finite_turn.set(finite)
    try:
        yield
    finally:
        _finite_turn.reset(token)


def admit_finite(params):
    if 'finite' not in params:
        return {}
    if type(params['finite']) is not bool:
        raise RuntimeStoreError('invalid_params')
    return {'finite': params['finite']}


async def execute_finite_admission(authority, ref, row):
    from gateway.session_ingress import execute_admission
    with finite_turn_scope(row['payload'].get('finite', False)):
        return await execute_admission(authority, ref, row)

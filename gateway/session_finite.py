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
    from gateway.session_surface import surface_turn_scope
    from gateway.session_hosted_output import (
        capture_failed_output, capture_output_result, hosted_output_scope,
    )
    with finite_turn_scope(row['payload'].get('finite', False)), \
            surface_turn_scope(row['payload'].get('surface_v1')), \
            hosted_output_scope(authority, ref, row) as output:
        try:
            response = await execute_admission(authority, ref, row)
            capture_output_result(authority, row, output)
            return response
        except Exception as exc:
            pending = capture_failed_output(authority, row, output)
            if pending is not None:
                exc.add_note(f"Group Chat output cleanup remains pending ({pending})")
            raise

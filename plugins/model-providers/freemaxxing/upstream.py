"""One free-only dispatch boundary for main, auxiliary, explicit, and MoA calls."""
from __future__ import annotations

from .policy import (AuthError, Budget, ModelNotFoundError,
                     _MAX_RESPONSE_BODY_BYTES, _is_router_model, guard_body)
from .pool import _refresh_backend_credentials, _resolve_auto_model, pool
from .protocol import loads, read_stream, validate_completion
from .transport import read_bounded


def _exhausted_message(last_error=None):
    return 'No eligible free route. Retry is safe for inference; no tool was executed by this router.'


def _open_response(backend, body, budget=None, *, owner=None):
    owner = owner or pool
    budget = budget or Budget(owner.limits.total)
    outgoing = dict(body)
    model = str(outgoing.get('model') or 'freemaxxing')
    if _is_router_model(model):
        model = _resolve_auto_model(backend)
    elif model.startswith(backend.name + '::'):
        model = model.split('::', 1)[1]
    outgoing['model'] = model

    def attempt(snapshot):
        with backend.refresh_lock:
            if backend.credential_snapshot() != snapshot:
                raise ModelNotFoundError('upstream credential changed before dispatch')
            row = backend.available_rows().get(model)
            if row is None:
                raise ModelNotFoundError('model has no current eligible free route')
            secured = guard_body(backend, outgoing, row)
        budget.remaining()
        return owner.transport.request(backend, 'POST', '/chat/completions', budget,
                                       body=secured, credentials=snapshot)

    observed = backend.credential_snapshot()
    if not observed[1] and backend.refresh is not None:
        _refresh_backend_credentials(backend, require_new=False, observed=observed)
        observed = backend.credential_snapshot()
    try:
        return attempt(observed)
    except AuthError:
        if not _refresh_backend_credentials(backend, require_new=True, observed=observed):
            raise
        # Network I/O is outside the credential lock. All waiters compare the
        # actual failed credential, not a later snapshot, so one refresh wins.
        # A new credential must earn its own catalog; never replay an old
        # account's price/capability grant after refreshing authentication.
        if not backend.catalog_lock.acquire(timeout=budget.remaining()):
            raise ModelNotFoundError('catalog refresh deadline exhausted')
        try:
            if model not in backend.available_rows():
                rows, credentials = owner._fetch_models(backend, budget)
                backend.set_catalog(rows, credentials=credentials)
        finally:
            backend.catalog_lock.release()
        return attempt(backend.credential_snapshot())


def _forward(backend, body, budget=None, *, owner=None):
    owner = owner or pool
    budget = budget or Budget(owner.limits.total)
    response = _open_response(backend, dict(body, stream=False), budget, owner=owner)
    return validate_completion(loads(read_bounded(response, budget, _MAX_RESPONSE_BODY_BYTES)))


def _open_stream(backend, body, budget=None, *, owner=None):
    owner = owner or pool
    budget = budget or Budget(owner.limits.total)
    response = _open_response(backend, dict(body, stream=True), budget, owner=owner)
    return read_stream(response, budget)

"""Main's ``gateway/session_worker.py`` bodies (as of the pinned main), the legacy side of the backend parity checks."""

import hmac
from hermes_state_runtime import _secret_digest
from persistence.core.errors import AuthorityRefusal


def _verify(connection, ref, params, claim):
    from hermes_state_worker_compression import worker_retry_target
    authority = connection.authority
    target = worker_retry_target(authority.db, params)
    with authority.db._read_ctx() as conn:
        row = conn.execute('SELECT * FROM worker_executions WHERE execution_id=?',
                           (params['execution_id'],)).fetchone()
        if row is None:
            raise AuthorityRefusal('not_found')
        if row['session_id'] != target:
            raise AuthorityRefusal('permission_denied')
        if type(params['generation']) is not int or row['generation'] != params['generation']:
            raise AuthorityRefusal('stale_generation')
        if not hmac.compare_digest(row['adoption_digest'], _secret_digest(claim)):
            raise AuthorityRefusal('permission_denied')
